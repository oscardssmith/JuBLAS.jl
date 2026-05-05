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

@testset "Kernel selection (multiple dispatch)" begin
    # Every kernel shape must produce the same result on Float64.
    M, N, K = 73, 65, 200   # exercises edge tiles in MR, NR, K
    A = randn(M, K); B = randn(K, N)
    ref = A * B

    kernels = (
        ScalarKernel{8,6}(),
        ScalarKernel{8,24}(),
        AVX512Kernel{8,6,Float64}(),
        AVX512Kernel{8,14,Float64}(),
        AVX512Kernel{8,24,Float64}(),
        AVX512Kernel{16,14,Float64}(),
    )
    for kernel in kernels
        C = zeros(M, N)
        gemm!(C, A, B; kernel=kernel)
        @test C ≈ ref
        C0 = randn(M, N); C = copy(C0)
        gemm!(C, A, B, -1.5, 0.5; kernel=kernel)
        @test C ≈ -1.5 .* ref .+ 0.5 .* C0
    end

    # Sanity: default for Float64 is the 8×24 AVX-512 kernel.
    @test default_kernel(Float64) === AVX512Kernel{8,24,Float64}()
    @test default_kernel(Float64) isa AVX512F64Kernel   # const alias still works
    @test default_kernel(Float32) isa ScalarKernel
end

@testset "gemm! performance (single-threaded vs OpenBLAS)" begin
    BLAS.set_num_threads(1)
    n = 512
    A = randn(n, n); B = randn(n, n); C = zeros(n, n)

    kernels = [
        ("Scalar 8x6",   ScalarKernel{8,6}()),
        ("AVX512 8x6",   AVX512Kernel{8,6, Float64}()),
        ("AVX512 8x14",  AVX512Kernel{8,14,Float64}()),
        ("AVX512 8x24",  AVX512Kernel{8,24,Float64}()),
        ("AVX512 16x14", AVX512Kernel{16,14,Float64}()),
    ]

    # warmup (also forces @generated body emission for each shape)
    for (_, k) in kernels; gemm!(C, A, B; kernel=k); end
    mul!(C, A, B)

    t_blas = @elapsed for _ in 1:5; mul!(C, A, B); end
    gflops_blas = 2 * n^3 * 5 / t_blas / 1e9

    for (name, k) in kernels
        t = @elapsed for _ in 1:5; gemm!(C, A, B; kernel=k); end
        gflops = 2 * n^3 * 5 / t / 1e9
        @printf("[%dx%d F64] %-13s: %6.1f GFLOPS   (OpenBLAS %6.1f GFLOPS, ratio %.2fx)\n",
                n, n, name, gflops, gflops_blas, t / t_blas)
    end
    # not asserted — informational
end
