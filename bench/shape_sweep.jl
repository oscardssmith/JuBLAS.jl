# Sweep kernel shapes (MR, NR) × problem shapes (M, N, K) and report
# GFLOPS. Used to verify whether the `default_kernel` choices win not
# just on the square case but also on common rectangular shapes
# (rank-k updates, tall-skinny C, wide C). Block sizes (MC, KC, NC) for
# each kernel come from `utils.jl`'s formulas.
#
# Usage:
#     julia --project=bench bench/shape_sweep.jl [N] [ITERS] [ELT]
# where ELT ∈ {Float32, Float64, ComplexF32, ComplexF64}. `N` is the
# baseline dimension; the rectangular shapes derive from it.

using JuBLAS, LinearAlgebra, Printf, Random

const N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
const ITERS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
const ELT   = length(ARGS) >= 3 ? Symbol(ARGS[3])     : :Float64

const T = eval(ELT)

const FLOPS_PER_MAC = T <: Complex ? 8 : 2

BLAS.set_num_threads(1)
Random.seed!(0xc0ffee)

# Problem shapes (M, N, K). C is M×N, A is M×K, B is K×N. Square is the
# baseline; the others stress one dimension at a time.
const SHAPES = [
    (N,    N,    N   ),    # square
    (N,    N,    N÷4),    # shallow K (rank-(N/4) update)
    (N,    N÷4, N   ),    # narrow C (few output cols)
    (N÷4, N,    N   ),    # short C (few output rows)
    (N÷2, N÷2, 4N  ),    # deep K (long inner dimension)
]

# Kernel shapes per eltype. Constraint: MR % W == 0 and total
# accumulator zmm registers fit within ~28 (32 zmms minus a few for A
# loads / B broadcast / temporaries).
#
#   Real cell = 1 zmm; budget ~28 → (MR/W) * NR ≤ 28.
#   Complex cell = 2 zmm (Re + Im accum); budget halved → (MR/W) * NR ≤ 14.
const KERNELS =
    T === Float64 ? [
        # rows = 1 (MR = W = 8): can chase wide NR.
        SIMDKernel{8,  8, 12, Float64}(),
        SIMDKernel{8,  8, 16, Float64}(),
        SIMDKernel{8,  8, 20, Float64}(),
        SIMDKernel{8,  8, 24, Float64}(),
        # rows = 2 (MR = 16): NR ≤ 14.
        SIMDKernel{8, 16,  6, Float64}(),
        SIMDKernel{8, 16, 10, Float64}(),
        SIMDKernel{8, 16, 12, Float64}(),
        SIMDKernel{8, 16, 14, Float64}(),    # current default
        # rows = 3 (MR = 24): NR ≤ 9.
        SIMDKernel{8, 24,  6, Float64}(),
        SIMDKernel{8, 24,  8, Float64}(),
    ] :
    T === Float32 ? [
        # rows = 1 (MR = 16): NR ≤ 28.
        SIMDKernel{16, 16, 12, Float32}(),
        SIMDKernel{16, 16, 16, Float32}(),
        SIMDKernel{16, 16, 20, Float32}(),
        SIMDKernel{16, 16, 24, Float32}(),
        # rows = 2 (MR = 32): NR ≤ 14.
        SIMDKernel{16, 32,  6, Float32}(),
        SIMDKernel{16, 32, 10, Float32}(),
        SIMDKernel{16, 32, 12, Float32}(),
        SIMDKernel{16, 32, 14, Float32}(),   # current default
        # rows = 3 (MR = 48): NR ≤ 9. Narrow-N candidate.
        SIMDKernel{16, 48,  6, Float32}(),
        SIMDKernel{16, 48,  8, Float32}(),
    ] :
    T === ComplexF64 ? [
        # rows = 1 (MR = 8): NR ≤ 14.
        SIMDKernel{8,  8,  6, ComplexF64}(),
        SIMDKernel{8,  8,  8, ComplexF64}(),
        SIMDKernel{8,  8, 10, ComplexF64}(),
        SIMDKernel{8,  8, 12, ComplexF64}(),
        SIMDKernel{8,  8, 14, ComplexF64}(),
        # rows = 2 (MR = 16): NR ≤ 7.
        SIMDKernel{8, 16,  4, ComplexF64}(),
        SIMDKernel{8, 16,  6, ComplexF64}(),  # current default
        # rows = 3 (MR = 24): NR ≤ 4. Narrow-N candidate.
        SIMDKernel{8, 24,  4, ComplexF64}(),
    ] :
    [
        # ComplexF32
        # rows = 1 (MR = 16): NR ≤ 14.
        SIMDKernel{16, 16,  6, ComplexF32}(),
        SIMDKernel{16, 16,  8, ComplexF32}(),
        SIMDKernel{16, 16, 10, ComplexF32}(),
        SIMDKernel{16, 16, 12, ComplexF32}(),
        SIMDKernel{16, 16, 14, ComplexF32}(),
        # rows = 2 (MR = 32): NR ≤ 7.
        SIMDKernel{16, 32,  4, ComplexF32}(),  # current default
        SIMDKernel{16, 32,  6, ComplexF32}(),
        # rows = 3 (MR = 48): NR ≤ 4. Narrow-N candidate.
        SIMDKernel{16, 48,  4, ComplexF32}(),
    ]

function bench_shape(C, A, B, kernel, iters)
    Apack, Bpack = JuBLAS.gemm_workspace(T, kernel)
    for _ in 1:5
        JuBLAS.gemm!(C, A, B; kernel=kernel, Apack=Apack, Bpack=Bpack)
    end
    GC.gc()
    t0 = time_ns()
    for _ in 1:iters
        JuBLAS.gemm!(C, A, B; kernel=kernel, Apack=Apack, Bpack=Bpack)
    end
    elapsed = (time_ns() - t0) / 1e9
    return elapsed
end

function bench_openblas(C, A, B, iters)
    for _ in 1:5; mul!(C, A, B); end
    GC.gc()
    t0 = time_ns()
    for _ in 1:iters; mul!(C, A, B); end
    elapsed = (time_ns() - t0) / 1e9
    return elapsed
end

# Thermal warmup so the first shape doesn't get a cool-CPU bonus.
let Awarm = randn(T, N, N), Bwarm = randn(T, N, N), Cwarm = zeros(T, N, N)
    t0 = time_ns()
    while (time_ns() - t0) < 3 * 1_000_000_000
        mul!(Cwarm, Awarm, Bwarm)
    end
end

_kernel_dims(::SIMDKernel{W,MR,NR,T}) where {W,MR,NR,T} = (W, MR, NR)
caches = JuBLAS._cache_sizes()

@printf("\n=== Kernel × shape sweep (T=%s, iters=%d) ===\n", T, ITERS)

for (M, Ncol, K) in SHAPES
    A = randn(T, M, K)
    B = randn(T, K, Ncol)
    C = zeros(T, M, Ncol)
    flops = FLOPS_PER_MAC * M * Ncol * K
    # Scale ITERS down for huge shapes so a single shape doesn't blow
    # the bench budget; the deep-K case has 4× the flop count.
    iters = max(div(ITERS * N^3, M * Ncol * K), 5)

    @printf("\n--- shape (M, N, K) = (%d, %d, %d)  iters=%d ---\n", M, Ncol, K, iters)
    @printf("%-44s  %5s  %5s  %5s  %4s  %8s  %8s\n",
            "kernel", "MR", "NR", "rows", "KC", "MC", "GFLOPS")
    for k in KERNELS
        W, MR, NR = _kernel_dims(k)
        KC = JuBLAS.kc_block(k, caches)
        MC = JuBLAS.mc_block(k, caches)
        rows = MR ÷ W
        elapsed = bench_shape(C, A, B, k, iters)
        gflops = flops * iters / elapsed / 1e9
        @printf("%-44s  %5d  %5d  %5d  %4d  %8d  %8.1f\n",
                string(typeof(k)), MR, NR, rows, KC, MC, gflops)
    end
    elapsed = bench_openblas(C, A, B, iters)
    gflops = flops * iters / elapsed / 1e9
    @printf("%-44s  %33s  %8.1f\n", "openblas (mul!)", "", gflops)
end
