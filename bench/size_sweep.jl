using JuBLAS, LinearAlgebra, Printf, Random, BenchmarkTools

# Sweep `gemm!` (with current `default_kernel`) vs OpenBLAS across the
# canonical shape list (see _shapes.jl). All shapes share thermal state
# (single process). Use this to spot regressions or wins anywhere on the
# curve, including rectangular cases with one small dim.
#
# Iteration count per shape is auto-tuned by `@belapsed`.
#
# Usage:
#     julia --project=bench bench/size_sweep.jl [ELT]
# ELT ∈ {Float32, Float64} (default Float32).

const ELT = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :Float32

const T = eval(ELT)
const FLOPS_PER_MAC = T <: Complex ? 8 : 2

include(joinpath(@__DIR__, "_shapes.jl"))

# Larger shapes need a bigger time budget so `@belapsed` gets enough samples;
# small ones need very little. Cap based on per-call work.
bench_budget(M, N, K) = max(0.2, min(1.0, 1e-9 * M * N * K * 200))

BLAS.set_num_threads(1)
Random.seed!(0xc0ffee)

let Awarm = randn(T, 256, 256), Bwarm = randn(T, 256, 256), Cwarm = zeros(T, 256, 256)
    t0 = time_ns()
    while (time_ns() - t0) < 3_000_000_000
        mul!(Cwarm, Awarm, Bwarm)
    end
end

@printf("\n=== gemm! vs OpenBLAS, default kernel (T=%s) ===\n", T)
@printf("%-15s  %5s  %5s  %5s  %12s  %12s  %9s\n",
        "label", "M", "N", "K", "ours GFLOPS", "OB GFLOPS", "ours/OB")

for (label, M, N, K) in SHAPES
    A = randn(T, M, K)
    B = randn(T, K, N)
    C = zeros(T, M, N)
    kernel = JuBLAS.default_kernel(T, M, N, K)
    Apack, Bpack = JuBLAS.gemm_workspace(T, kernel)
    budget = bench_budget(M, N, K)

    t_ours = @belapsed JuBLAS.gemm!($C, $A, $B; kernel=$kernel,
                                    Apack=$Apack, Bpack=$Bpack) seconds=budget
    t_blas = @belapsed mul!($C, $A, $B) seconds=budget

    flops  = FLOPS_PER_MAC * Float64(M) * Float64(N) * Float64(K)
    ours_g = flops / t_ours / 1e9
    blas_g = flops / t_blas / 1e9
    @printf("%-15s  %5d  %5d  %5d  %12.1f  %12.1f  %9.2f\n",
            label, M, N, K, ours_g, blas_g, ours_g / blas_g)
end
