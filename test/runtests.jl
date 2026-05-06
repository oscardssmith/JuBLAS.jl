using BenchmarkTools, Test, LinearAlgebra, Random, Printf, JuBLAS

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

    # `default_kernel` is now CPU-dependent (picked from detected SIMD width).
    expected_F64 =
        JuBLAS._simd_bytes() >= 64 ? SIMDKernel{8, 16, 14, Float64}() :
        JuBLAS._simd_bytes() >= 32 ? SIMDKernel{4,  8,  6, Float64}() :
        JuBLAS._simd_bytes() >= 16 ? SIMDKernel{2,  2,  4, Float64}() :
                                      ScalarKernel{8, 6}()
    @test default_kernel(Float64) === expected_F64
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

    expected_F32 =
        JuBLAS._simd_bytes() >= 64 ? SIMDKernel{16, 32, 14, Float32}() :
        JuBLAS._simd_bytes() >= 32 ? SIMDKernel{8,  16,  6, Float32}() :
        JuBLAS._simd_bytes() >= 16 ? SIMDKernel{4,   4,  6, Float32}() :
                                      ScalarKernel{16, 6}()
    @test default_kernel(Float32) === expected_F32
end

@testset "Kernel selection (Complex)" begin
    # Sizes chosen so partial-tile and partial-row-group writebacks fire:
    # - M=33: with MR=8 → mr_=1 on the last ir block; with MR=16 (rows=2) → mr_=1
    #   (only first row-group active, second row-group has mr_-W ≤ 0).
    # - N=25: with NR=12 → nr_=1 on the last jr block.
    # - K=50: doesn't affect edge dispatch but stresses the kc accumulation.
    M, N, K = 33, 25, 50

    for T in (ComplexF64, ComplexF32)
        TR = real(T)
        A = randn(T, M, K); B = randn(T, K, N)
        ref = A * B
        rtol = TR === Float32 ? 1f-4 : 1e-10

        kernels = T === ComplexF64 ? (
            SIMDKernel{8,  8, 12, ComplexF64}(),    # AVX-512 zmm, 8×12
            SIMDKernel{8, 16, 12, ComplexF64}(),    # AVX-512 zmm, rows=2
            SIMDKernel{4,  4,  6, ComplexF64}(),    # AVX2 ymm
            SIMDKernel{4,  8,  6, ComplexF64}(),    # AVX2 ymm, rows=2
        ) : (
            SIMDKernel{16, 16, 12, ComplexF32}(),   # AVX-512 zmm
            SIMDKernel{16, 32, 12, ComplexF32}(),   # AVX-512 zmm, rows=2
            SIMDKernel{ 8,  8,  6, ComplexF32}(),   # AVX2 ymm
            SIMDKernel{ 8, 16,  6, ComplexF32}(),   # AVX2 ymm, rows=2
        )

        for kernel in kernels
            C = zeros(T, M, N)
            gemm!(C, A, B; kernel=kernel)
            @test C ≈ ref rtol=rtol

            C0 = randn(T, M, N); C = copy(C0)
            α = TR(1.5) - TR(0.3)*im
            β = TR(0.7) + TR(0.2)*im
            gemm!(C, A, B, α, β; kernel=kernel)
            @test C ≈ α .* ref .+ β .* C0  rtol=rtol
        end

        expected = TR === Float64 ? (
            JuBLAS._simd_bytes() >= 64 ? SIMDKernel{ 8, 16,  6, ComplexF64}() :
            JuBLAS._simd_bytes() >= 32 ? SIMDKernel{ 4,  4,  6, ComplexF64}() :
                                          SIMDKernel{ 2,  2,  4, ComplexF64}()
        ) : (
            JuBLAS._simd_bytes() >= 64 ? SIMDKernel{16, 32,  4, ComplexF32}() :
            JuBLAS._simd_bytes() >= 32 ? SIMDKernel{ 8,  8,  6, ComplexF32}() :
                                          SIMDKernel{ 4,  4,  6, ComplexF32}()
        )
        @test default_kernel(T) === expected
    end
end
