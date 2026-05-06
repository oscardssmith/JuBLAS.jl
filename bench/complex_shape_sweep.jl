# Sweep complex kernel shapes (MR, NR) and report GFLOPS. Used to verify
# whether the default `default_complex_kernel` choices (8×12 for F64,
# 16×12 for F32) are actually winning on this host, or if a different
# tile shape would be faster.
#
# Block sizes (MC, KC, NC) for each kernel come from `utils.jl`'s formulas.

using JuBLAS, LinearAlgebra, Printf, Random

const N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
const ITERS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
const ELT   = length(ARGS) >= 3 ? Symbol(ARGS[3])     : :ComplexF64

BLAS.set_num_threads(1)

const T = ELT === :ComplexF32 ? ComplexF32 : ComplexF64
const TR = T === ComplexF64 ? Float64 : Float32

Random.seed!(0xc0ffee)
const A = randn(T, N, N)
const B = randn(T, N, N)
const C = zeros(T, N, N)

# Shapes to compare. Constraint: MR % W == 0; total accumulator zmms =
# (MR/W) * NR * 2 ≤ ~28 to leave room for A loads + temporaries.
#
# Float64 / W=8 budget:
#   rows=1: NR ≤ 14 (28 / 2)
#   rows=2 (MR=16): NR ≤ 6 (12 cells × 2 = 24 zmm + 4 Are/Aim = 28)
#
# Float32 / W=16 budget: same shape constraints but tile holds 2× elements.

const SHAPES = T === ComplexF64 ? [
    # rows = 1
    SIMDKernel{8,  8,  6, ComplexF64}(),
    SIMDKernel{8,  8,  8, ComplexF64}(),
    SIMDKernel{8,  8, 10, ComplexF64}(),
    SIMDKernel{8,  8, 12, ComplexF64}(),    # current default
    SIMDKernel{8,  8, 14, ComplexF64}(),
    # rows = 2
    SIMDKernel{8, 16,  4, ComplexF64}(),
    SIMDKernel{8, 16,  6, ComplexF64}(),
] : [
    # rows = 1
    SIMDKernel{16, 16,  6, ComplexF32}(),
    SIMDKernel{16, 16,  8, ComplexF32}(),
    SIMDKernel{16, 16, 10, ComplexF32}(),
    SIMDKernel{16, 16, 12, ComplexF32}(),   # current default
    SIMDKernel{16, 16, 14, ComplexF32}(),
    # rows = 2
    SIMDKernel{16, 32,  4, ComplexF32}(),
    SIMDKernel{16, 32,  6, ComplexF32}(),
]

function bench_shape(kernel)
    Apack, Bpack = JuBLAS.gemm_workspace(T, kernel)
    # Warm up
    for _ in 1:5
        JuBLAS.gemm!(C, A, B; kernel=kernel, Apack=Apack, Bpack=Bpack)
    end
    GC.gc()
    t0 = time_ns()
    for _ in 1:ITERS
        JuBLAS.gemm!(C, A, B; kernel=kernel, Apack=Apack, Bpack=Bpack)
    end
    elapsed = (time_ns() - t0) / 1e9
    gflops = 8 * N^3 * ITERS / elapsed / 1e9
    return elapsed, gflops
end

# OpenBLAS reference for context
function bench_openblas()
    for _ in 1:5; mul!(C, A, B); end
    GC.gc()
    t0 = time_ns()
    for _ in 1:ITERS; mul!(C, A, B); end
    elapsed = (time_ns() - t0) / 1e9
    gflops = 8 * N^3 * ITERS / elapsed / 1e9
    return elapsed, gflops
end

@printf("\n=== Complex kernel-shape sweep (T=%s, N=%d, iters=%d) ===\n", T, N, ITERS)
@printf("%-44s  %5s  %5s  %5s  %4s  %8s  %8s\n",
        "kernel", "MR", "NR", "rows", "KC", "MC", "GFLOPS")

_kernel_dims(::SIMDKernel{W,MR,NR,T}) where {W,MR,NR,T} = (W, MR, NR)

caches = JuBLAS._cache_sizes()
for k in SHAPES
    W, MR, NR = _kernel_dims(k)
    KC = JuBLAS.kc_block(k, caches)
    MC = JuBLAS.mc_block(k, caches)
    rows = MR ÷ W
    elapsed, gflops = bench_shape(k)
    @printf("%-44s  %5d  %5d  %5d  %4d  %8d  %8.1f\n",
            string(typeof(k)), MR, NR, rows, KC, MC, gflops)
end

elapsed, gflops = bench_openblas()
@printf("%-44s  %33s  %8.1f\n", "openblas (mul!)", "", gflops)
