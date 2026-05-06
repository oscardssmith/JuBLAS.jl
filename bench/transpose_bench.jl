using JuBLAS, LinearAlgebra, LinuxPerf, Printf, Random

# Sweeps `gemm!` perf vs OpenBLAS / MKL across every (opA, opB) ∈
# {N, T, C}² combo, for any supported eltype. Correctness should be
# verified separately via `test/transpose_check.jl`.
#
# Usage:
#     julia --project=bench bench/transpose_bench.jl [N] [ITERS] [ELT]
# where ELT ∈ {Float32, Float64, ComplexF32, ComplexF64}.

const N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
const ITERS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5000
const ELT   = length(ARGS) >= 3 ? Symbol(ARGS[3])     : :Float32

const T = ELT === :Float32    ? Float32    :
          ELT === :Float64    ? Float64    :
          ELT === :ComplexF32 ? ComplexF32 :
          ELT === :ComplexF64 ? ComplexF64 :
          error("unknown eltype: $ELT  (expected Float32, Float64, ComplexF32, ComplexF64)")

# Real gemm: 2 flops per output-elt-per-K (1 mul + 1 add).
# Complex: 8 flops per output-elt-per-K (4 muls + 4 adds for one cMAC).
const FLOPS_PER_MAC = T <: Complex ? 8 : 2

BLAS.set_num_threads(1)
Random.seed!(0xc0ffee)

apply_op(A, op::Symbol) = op === :N ? A :
                          op === :T ? transpose(A) :
                          adjoint(A)

# Square problem so a single ITERS budget covers every combo at the same
# flop count. Returns `(parent_storage, view_used_in_gemm)`; parent must
# stay rooted because `transpose`/`adjoint` are lazy wrappers.
function make_arg(::Type{T}, op::Symbol, rows::Int, cols::Int) where {T}
    if op === :N
        P = randn(T, rows, cols)
        return P, P
    else
        P = randn(T, cols, rows)
        return P, apply_op(P, op)
    end
end

const C = zeros(T, N, N)
const KERNEL = JuBLAS.default_kernel(T)
const Apack, Bpack = JuBLAS.gemm_workspace(T, KERNEL)

const EVENTS = "(cpu-cycles,instructions,branch-instructions,branch-misses)," *
               "(L1-dcache-loads,L1-dcache-load-misses)," *
               "(LLC-loads,LLC-load-misses)"

function bench(label, work)
    GC.gc()
    t0 = time_ns()
    stats = @pstats EVENTS work()
    elapsed = (time_ns() - t0) / 1e9
    gflops = FLOPS_PER_MAC * N^3 * ITERS / elapsed / 1e9
    @printf("\n=== %s  —  %.2fs  %.1f GFLOPS  (N=%d, iters=%d, T=%s) ===\n",
            label, elapsed, gflops, N, ITERS, T)
    show(stdout, MIME"text/plain"(), stats)
    println()
end

const COMBOS = [(opA, opB) for opA in (:N, :T, :C), opB in (:N, :T, :C)][:]

# Thermal conditioning: drive the CPU to its steady-state DVFS regime
# before any combo is timed, so the first one isn't measured on a
# cool/over-clocked CPU and the last on a thermally-throttled one. 3s of
# OpenBLAS mul! on this shape gets the package over its boost-clock
# transient and into sustained-load frequency.
let Awarm = randn(T, N, N), Bwarm = randn(T, N, N), Cwarm = zeros(T, N, N)
    t0 = time_ns()
    while (time_ns() - t0) < 3 * 1_000_000_000
        mul!(Cwarm, Awarm, Bwarm)
    end
end

for (opA, opB) in COMBOS
    PA, A = make_arg(T, opA, N, N)
    PB, B = make_arg(T, opB, N, N)

    ours_loop() = (for _ in 1:ITERS
        JuBLAS.gemm!(C, A, B; kernel=KERNEL, Apack=Apack, Bpack=Bpack)
    end)
    mul_loop()  = (for _ in 1:ITERS; mul!(C, A, B); end)

    for _ in 1:5
        JuBLAS.gemm!(C, A, B; kernel=KERNEL, Apack=Apack, Bpack=Bpack)
        mul!(C, A, B)
    end
    bench("ours      op(A)=$opA op(B)=$opB", ours_loop)
    bench("openblas  op(A)=$opA op(B)=$opB", mul_loop)
end

@info "switching BLAS backend to MKL"
using MKL
BLAS.set_num_threads(1)
@info "active BLAS: $(BLAS.get_config())"

for (opA, opB) in COMBOS
    PA, A = make_arg(T, opA, N, N)
    PB, B = make_arg(T, opB, N, N)
    mul_loop() = (for _ in 1:ITERS; mul!(C, A, B); end)
    for _ in 1:5; mul!(C, A, B); end
    bench("mkl       op(A)=$opA op(B)=$opB", mul_loop)
end
