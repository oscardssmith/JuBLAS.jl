using Test, LinearAlgebra, Random, Printf, JuBLAS

Random.seed!(0)

@testset "gemm! correctness" begin
    @testset "Float64 size sweep" begin
        # Cover: tiny, exact-tile, single edge, multi-block, both-edges,
        # M-edge only, N-edge only, K not block-multiple.
        sizes = [(1,1,1), (8,6,256), (10,7,9), (72,4080,256),
                 (73,6,257), (200,200,200), (5,128,3), (128,5,257),
                 (16,12,512), (100,100,100)]
        for (M,N,K) in sizes
            A = randn(M,K); B = randn(K,N); C = randn(M,N)
            ref = 2.5 .* (A*B) .+ 1.5 .* C
            gemm!(C, A, B, 2.5, 1.5)
            err = maximum(abs, C .- ref) / max(1.0, maximum(abs, ref))
            @test err < 1e-10
        end
    end

    @testset "α and β edge values" begin
        M, N, K = 64, 48, 80
        A = randn(M,K); B = randn(K,N)
        # β=0 (default): C is overwritten regardless of prior contents (incl. NaN)
        C = fill(NaN, M, N)
        gemm!(C, A, B)
        @test C ≈ A*B
        # α=1, β=1: pure accumulate
        C = randn(M, N); C0 = copy(C)
        gemm!(C, A, B, true, true)
        @test C ≈ C0 .+ A*B
        # α negative, β fractional
        C = randn(M, N); C0 = copy(C)
        gemm!(C, A, B, -2.0, 0.25)
        @test C ≈ -2.0 .* (A*B) .+ 0.25 .* C0
    end

    @testset "Float32" begin
        A = randn(Float32, 64, 80)
        B = randn(Float32, 80, 48)
        C = zeros(Float32, 64, 48)
        gemm!(C, A, B)
        @test C ≈ A*B rtol=1f-4
    end

    @testset "ComplexF64" begin
        A = randn(ComplexF64, 32, 40)
        B = randn(ComplexF64, 40, 24)
        C = zeros(ComplexF64, 32, 24)
        gemm!(C, A, B)
        @test C ≈ A*B
        # with α, β complex
        C0 = randn(ComplexF64, 32, 24); C = copy(C0)
        α = 1.5 - 0.3im; β = 0.7 + 0.2im
        gemm!(C, A, B, α, β)
        @test C ≈ α .* (A*B) .+ β .* C0
    end

    @testset "Empty dimensions" begin
        @test gemm!(zeros(0, 5), zeros(0, 3), zeros(3, 5)) == zeros(0, 5)
        @test gemm!(zeros(5, 0), zeros(5, 3), zeros(3, 0)) == zeros(5, 0)
        # K=0: result is β·C
        C = randn(4, 4); C0 = copy(C)
        gemm!(C, zeros(4, 0), zeros(0, 4), 1.0, 2.0)
        @test C ≈ 2.0 .* C0
    end

    @testset "Dimension mismatch throws" begin
        @test_throws DimensionMismatch gemm!(zeros(4,4), zeros(5,4), zeros(4,4))
        @test_throws DimensionMismatch gemm!(zeros(4,4), zeros(4,4), zeros(4,5))
        @test_throws DimensionMismatch gemm!(zeros(4,4), zeros(4,3), zeros(4,4))
    end

    @testset "Views (non-contiguous inputs)" begin
        Afull = randn(100, 100); Bfull = randn(100, 100); Cfull = randn(100, 100)
        A = view(Afull, 10:80, 5:90)
        B = view(Bfull, 5:90, 20:75)
        C = copy(view(Cfull, 10:80, 20:75))
        ref = A*B
        gemm!(C, A, B)
        @test C ≈ ref
    end
end

@testset "Kernel selection (Float64)" begin
    M, N, K = 73, 65, 200   # exercises edge tiles in MR, NR, K
    A = randn(M, K); B = randn(K, N)
    ref = A * B

    kernels = (
        ScalarKernel{8,6}(),
        ScalarKernel{8,24}(),
        SIMDKernel{8, 8, 6,Float64}(),
        SIMDKernel{8, 8,14,Float64}(),
        SIMDKernel{8, 8,24,Float64}(),
        SIMDKernel{8,16,14,Float64}(),
        SIMDKernel{4, 4, 6,Float64}(),    # AVX2 ymm
        SIMDKernel{4, 8, 6,Float64}(),    # AVX2 ymm, 2 ymm/col
    )
    for kernel in kernels
        C = zeros(M, N)
        gemm!(C, A, B; kernel=kernel)
        @test C ≈ ref
        C0 = randn(M, N); C = copy(C0)
        gemm!(C, A, B, -1.5, 0.5; kernel=kernel)
        @test C ≈ -1.5 .* ref .+ 0.5 .* C0
    end

    @test default_kernel(Float64) === SIMDKernel{8,8,24,Float64}()
end

@testset "Kernel selection (Float32)" begin
    # Sizes chosen to hit edges for the wider Float32 SIMD shapes (W=16).
    M, N, K = 81, 73, 200
    A = randn(Float32, M, K); B = randn(Float32, K, N)
    ref = A * B

    kernels = (
        ScalarKernel{16,6}(),
        SIMDKernel{16,16, 6,Float32}(),    # AVX-512 zmm, 16×6
        SIMDKernel{16,16,14,Float32}(),    # AVX-512 zmm, 16×14
        SIMDKernel{16,16,24,Float32}(),    # AVX-512 zmm, 16×24 (default)
        SIMDKernel{16,32,14,Float32}(),    # AVX-512 zmm, 32×14 (2 zmm/col)
        SIMDKernel{ 8, 8, 6,Float32}(),    # AVX2 ymm, 8×6
        SIMDKernel{ 8,16, 6,Float32}(),    # AVX2 ymm, 16×6 (2 ymm/col)
    )
    for kernel in kernels
        C = zeros(Float32, M, N)
        gemm!(C, A, B; kernel=kernel)
        @test C ≈ ref rtol=1f-4
        C0 = randn(Float32, M, N); C = copy(C0)
        gemm!(C, A, B, -1.5f0, 0.5f0; kernel=kernel)
        @test C ≈ -1.5f0 .* ref .+ 0.5f0 .* C0 rtol=1f-4
    end

    @test default_kernel(Float32) === SIMDKernel{16,16,24,Float32}()
end

function _bench(label::AbstractString, ::Type{T}, n::Int, kernels) where {T}
    BLAS.set_num_threads(1)
    A = randn(T, n, n); B = randn(T, n, n); C = zeros(T, n, n)

    # warmup (also forces @generated body emission for each shape)
    for (_, k) in kernels; gemm!(C, A, B; kernel=k); end
    mul!(C, A, B)

    t_blas = @elapsed for _ in 1:5; mul!(C, A, B); end
    gflops_blas = 2 * n^3 * 5 / t_blas / 1e9

    for (name, k) in kernels
        t = @elapsed for _ in 1:5; gemm!(C, A, B; kernel=k); end
        gflops = 2 * n^3 * 5 / t / 1e9
        @printf("[%dx%d %s] %-28s: %6.1f GFLOPS   (OpenBLAS %6.1f GFLOPS, ratio %.2fx)\n",
                n, n, label, name, gflops, gflops_blas, t / t_blas)
    end
end

@testset "gemm! Float64 performance (single-threaded vs OpenBLAS)" begin
    kernels = [
        ("Scalar 8x6",            ScalarKernel{8,6}()),
        ("SIMD W=4 (AVX2)   4x6", SIMDKernel{4, 4, 6,Float64}()),
        ("SIMD W=4 (AVX2)   8x6", SIMDKernel{4, 8, 6,Float64}()),
        ("SIMD W=8 (AVX512) 8x6", SIMDKernel{8, 8, 6,Float64}()),
        ("SIMD W=8 (AVX512) 8x14",SIMDKernel{8, 8,14,Float64}()),
        ("SIMD W=8 (AVX512) 8x24",SIMDKernel{8, 8,24,Float64}()),
        ("SIMD W=8 (AVX512)16x14",SIMDKernel{8,16,14,Float64}()),
    ]
    _bench("F64", Float64, 512, kernels)
    # not asserted — informational
end

@testset "gemm! Float32 performance (single-threaded vs OpenBLAS)" begin
    kernels = [
        ("Scalar 16x6",             ScalarKernel{16,6}()),
        ("SIMD W=8  (AVX2)    8x6", SIMDKernel{ 8, 8, 6,Float32}()),
        ("SIMD W=8  (AVX2)   16x6", SIMDKernel{ 8,16, 6,Float32}()),
        ("SIMD W=16 (AVX512) 16x6", SIMDKernel{16,16, 6,Float32}()),
        ("SIMD W=16 (AVX512) 16x14",SIMDKernel{16,16,14,Float32}()),
        ("SIMD W=16 (AVX512) 16x24",SIMDKernel{16,16,24,Float32}()),
        ("SIMD W=16 (AVX512) 32x14",SIMDKernel{16,32,14,Float32}()),
    ]
    _bench("F32", Float32, 512, kernels)
    # not asserted — informational
end
