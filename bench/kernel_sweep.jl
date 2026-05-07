using JuBLAS, LinearAlgebra, Printf, Random, BenchmarkTools
using JuBLAS: SIMDKernel

# Sweep candidate kernel shapes (MR×NR) across the canonical shape list
# (see _shapes.jl). For each shape, every kernel candidate is benched and
# the best per shape is reported in the summary. Goal: produce data so
# `default_kernel(T, M, N, K)` can route on the full shape.
#
# Iteration count is auto-tuned by `@belapsed`. No manual iters per shape.
#
# Usage:
#     julia --project=bench bench/kernel_sweep.jl [ELT]
# ELT ∈ {Float32, Float64}. All shapes share thermal state (single process).

const ELT = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :Float32

const T = eval(ELT)
const FLOPS_PER_MAC = T <: Complex ? 8 : 2

include(joinpath(@__DIR__, "_shapes.jl"))

# Kernel candidates dispatched on detected SIMD width. AVX-512 (zmm, 64 B):
# real cell = 1 zmm, budget ~28; complex cell = 2 zmm, budget ~14. AVX2
# (ymm, 32 B): 16 ymm registers, real budget ~14, complex budget ~7.
# To tune the AVX2 path on AVX-512 hardware:
#     JULIA_JUBLAS_SIMD_BYTES=32 julia --cpu-target=haswell \
#         --project=bench bench/kernel_sweep.jl Float32
const SB = JuBLAS._simd_bytes()

const KERNELS = T === Float32 ? (SB >= 64 ? [
    # AVX-512 Float32: W = 16.
    SIMDKernel{16, 16,  4, Float32}(),
    SIMDKernel{16, 16,  6, Float32}(),
    SIMDKernel{16, 16,  8, Float32}(),
    SIMDKernel{16, 16, 12, Float32}(),
    SIMDKernel{16, 16, 16, Float32}(),
    SIMDKernel{16, 16, 20, Float32}(),
    SIMDKernel{16, 16, 24, Float32}(),
    SIMDKernel{16, 32,  4, Float32}(),
    SIMDKernel{16, 32,  6, Float32}(),
    SIMDKernel{16, 32,  8, Float32}(),
    SIMDKernel{16, 32, 10, Float32}(),
    SIMDKernel{16, 32, 12, Float32}(),
    SIMDKernel{16, 32, 14, Float32}(),
] : [
    # AVX2 Float32: W = 8. Budget (MR/W)*NR ≤ 14.
    SIMDKernel{8,  8,  4, Float32}(),
    SIMDKernel{8,  8,  6, Float32}(),
    SIMDKernel{8,  8,  8, Float32}(),
    SIMDKernel{8,  8, 12, Float32}(),
    SIMDKernel{8,  8, 14, Float32}(),
    SIMDKernel{8, 16,  4, Float32}(),
    SIMDKernel{8, 16,  6, Float32}(),
    SIMDKernel{8, 16,  7, Float32}(),
    SIMDKernel{8, 24,  4, Float32}(),
]) : T === Float64 ? (SB >= 64 ? [
    # AVX-512 Float64: W = 8.
    SIMDKernel{8,  8,  4, Float64}(),
    SIMDKernel{8,  8,  6, Float64}(),
    SIMDKernel{8,  8,  8, Float64}(),
    SIMDKernel{8,  8, 12, Float64}(),
    SIMDKernel{8,  8, 16, Float64}(),
    SIMDKernel{8,  8, 20, Float64}(),
    SIMDKernel{8,  8, 24, Float64}(),
    SIMDKernel{8,  8, 28, Float64}(),
    SIMDKernel{8, 16,  4, Float64}(),
    SIMDKernel{8, 16,  6, Float64}(),
    SIMDKernel{8, 16,  8, Float64}(),
    SIMDKernel{8, 16, 12, Float64}(),
    SIMDKernel{8, 16, 14, Float64}(),
    SIMDKernel{8, 24,  4, Float64}(),
    SIMDKernel{8, 24,  6, Float64}(),
    SIMDKernel{8, 24,  8, Float64}(),
] : [
    # AVX2 Float64: W = 4. Budget (MR/W)*NR ≤ 14.
    SIMDKernel{4,  4,  4, Float64}(),
    SIMDKernel{4,  4,  6, Float64}(),
    SIMDKernel{4,  4,  8, Float64}(),
    SIMDKernel{4,  4, 12, Float64}(),
    SIMDKernel{4,  4, 14, Float64}(),
    SIMDKernel{4,  8,  4, Float64}(),
    SIMDKernel{4,  8,  6, Float64}(),
    SIMDKernel{4,  8,  7, Float64}(),
    SIMDKernel{4, 12,  4, Float64}(),
]) : T === ComplexF32 ? (SB >= 64 ? [
    # AVX-512 ComplexF32: W = 16. Complex budget (MR/W)*NR ≤ 14.
    SIMDKernel{16, 16,  4, ComplexF32}(),
    SIMDKernel{16, 16,  6, ComplexF32}(),
    SIMDKernel{16, 16,  8, ComplexF32}(),
    SIMDKernel{16, 16, 10, ComplexF32}(),
    SIMDKernel{16, 16, 12, ComplexF32}(),
    SIMDKernel{16, 16, 14, ComplexF32}(),
    SIMDKernel{16, 32,  4, ComplexF32}(),
    SIMDKernel{16, 32,  6, ComplexF32}(),
    SIMDKernel{16, 32,  7, ComplexF32}(),
] : [
    # AVX2 ComplexF32: W = 8. Complex budget (MR/W)*NR ≤ 7.
    SIMDKernel{8,  8,  4, ComplexF32}(),
    SIMDKernel{8,  8,  6, ComplexF32}(),
    SIMDKernel{8,  8,  7, ComplexF32}(),
    SIMDKernel{8, 16,  4, ComplexF32}(),
]) : (SB >= 64 ? [
    # AVX-512 ComplexF64: W = 8. Complex budget ≤ 14.
    SIMDKernel{8,  8,  4, ComplexF64}(),
    SIMDKernel{8,  8,  6, ComplexF64}(),
    SIMDKernel{8,  8,  8, ComplexF64}(),
    SIMDKernel{8,  8, 10, ComplexF64}(),
    SIMDKernel{8,  8, 12, ComplexF64}(),
    SIMDKernel{8,  8, 14, ComplexF64}(),
    SIMDKernel{8, 16,  4, ComplexF64}(),
    SIMDKernel{8, 16,  6, ComplexF64}(),
    SIMDKernel{8, 16,  7, ComplexF64}(),
] : [
    # AVX2 ComplexF64: W = 4. Complex budget ≤ 7.
    SIMDKernel{4,  4,  4, ComplexF64}(),
    SIMDKernel{4,  4,  6, ComplexF64}(),
    SIMDKernel{4,  4,  7, ComplexF64}(),
    SIMDKernel{4,  8,  4, ComplexF64}(),
])

BLAS.set_num_threads(1)
Random.seed!(0xc0ffee)

# Cap per-call budget so the whole sweep finishes in reasonable wall time.
# 0.3s per cell × ~14 shapes × ~13 kernels ≈ 1 minute per eltype.
const BENCH_TIME = 0.3

let Awarm = randn(T, 256, 256), Bwarm = randn(T, 256, 256), Cwarm = zeros(T, 256, 256)
    t0 = time_ns()
    while (time_ns() - t0) < 3_000_000_000
        mul!(Cwarm, Awarm, Bwarm)
    end
end

function bench_kernel(C, A, B, kernel)
    Apack, Bpack = JuBLAS.gemm_workspace(T, kernel)
    @belapsed JuBLAS.gemm!($C, $A, $B; kernel=$kernel, Apack=$Apack, Bpack=$Bpack) seconds=BENCH_TIME
end

bench_openblas(C, A, B) =
    @belapsed mul!($C, $A, $B) seconds=BENCH_TIME

_dims(::SIMDKernel{W,MR,NR,T}) where {W,MR,NR,T} = (W, MR, NR)

@printf("\n=== kernel sweep (T=%s) ===\n", T)

best_by_shape = Tuple{String,Int,Int,Int,String,Float64,Float64}[]

for (label, M, N, K) in SHAPES
    A = randn(T, M, K)
    B = randn(T, K, N)
    C = zeros(T, M, N)
    flops = FLOPS_PER_MAC * Float64(M) * Float64(N) * Float64(K)

    @printf("\n--- %-15s  (M, N, K) = (%d, %d, %d) ---\n", label, M, N, K)
    @printf("%-32s  %4s  %4s  %10s  %8s\n", "kernel", "MR", "NR", "GFLOPS", "vs OB")

    t_blas = bench_openblas(C, A, B)
    blas_g = flops / t_blas / 1e9

    best_g   = 0.0
    best_str = ""
    for kernel in KERNELS
        W, MR, NR = _dims(kernel)
        elapsed = bench_kernel(C, A, B, kernel)
        gflops  = flops / elapsed / 1e9
        @printf("%-32s  %4d  %4d  %10.1f  %7.2f\n",
                string(typeof(kernel)), MR, NR, gflops, gflops / blas_g)
        if gflops > best_g
            best_g   = gflops
            best_str = "MR=$MR NR=$NR"
        end
    end
    @printf("%-32s  %16s  %10.1f\n", "openblas mul!", "", blas_g)
    push!(best_by_shape, (label, M, N, K, best_str, best_g, blas_g))
end

@printf("\n=== best kernel per shape ===\n")
@printf("%-15s  %5s  %5s  %5s  %-20s  %10s  %10s  %8s\n",
        "label", "M", "N", "K", "best", "GFLOPS", "OB GFLOPS", "ours/OB")
for (label, M, N, K, best_str, best_g, blas_g) in best_by_shape
    @printf("%-15s  %5d  %5d  %5d  %-20s  %10.1f  %10.1f  %8.2f\n",
            label, M, N, K, best_str, best_g, blas_g, best_g / blas_g)
end
