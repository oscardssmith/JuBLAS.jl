# Reference-BLAS (dgemm/zgemm) α/β semantics:
#   - α == 0: C ← β·C and A, B are never read, so NaN/Inf in A or B must
#     not reach C;
#   - β == 0: C is never read, so NaN/Inf in C is overwritten;
#   - M == 0 or N == 0: no-op.
# K == 0 gives exactly β·C for any α (where reference BLAS's transposed-A
# loops would give α·0, i.e. NaN for non-finite α).

using JuBLAS, LinearAlgebra, Random, Test

Random.seed!(1)

# Random matrix with NaN/Inf sprinkled in (including complex NaN/Inf mixes).
function poisoned(::Type{T}, m, n) where {T}
    X = randn(T, m, n)
    X[1:3:end] .= T(NaN)
    X[2:7:end] .= T <: Complex ? T(Inf, NaN) : T(Inf)
    X[3:11:end] .= T(-Inf)
    return X
end

alpha_zeros(::Type{T}) where {T} =
    T <: Complex ? (zero(T), false, T(-0.0, 0.0)) : (zero(T), false, -zero(T))
nonzero_betas(::Type{T}) where {T} =
    T <: Complex ? (one(T), T(-0.5), T(0.5, -2.0), true) : (one(T), T(-0.5), T(3), true)

function check_alpha_beta_semantics(::Type{T}, M, N, K; kw...) where {T}
    A = poisoned(T, M, K)
    B = poisoned(T, K, N)
    # α == 0, β ≠ 0: exactly β·C.
    for α in alpha_zeros(T), β in nonzero_betas(T)
        C0 = randn(T, M, N)
        C  = copy(C0)
        gemm!(C, A, B, α, β; kw...)
        @test C == T(β) .* C0
    end
    # α == 0, β == 0: exactly zero even with NaN/Inf in A, B and C.
    for α in alpha_zeros(T), β in (zero(T), false)
        C = poisoned(T, M, N)
        gemm!(C, A, B, α, β; kw...)
        @test all(iszero, C)
    end
    # β == 0, α ≠ 0: NaN/Inf in C is overwritten, not propagated.
    A = randn(T, M, K)
    B = randn(T, K, N)
    C = poisoned(T, M, N)
    gemm!(C, A, B, T(2), false; kw...)
    @test all(isfinite, C)
    @test C ≈ 2 .* (A * B)
end

# Operand whose elements must never be read.
struct Unreadable{T} <: AbstractMatrix{T}
    m::Int
    n::Int
end
Base.size(X::Unreadable) = (X.m, X.n)
Base.getindex(::Unreadable, ::Int, ::Int) = error("operand was read")

# C that may be written but not read.
struct WriteOnly{T} <: AbstractMatrix{T}
    data::Matrix{T}
end
Base.size(X::WriteOnly) = size(X.data)
Base.getindex(::WriteOnly, ::Int, ::Int) = error("C was read")
Base.setindex!(X::WriteOnly, v, i::Int, j::Int) = (X.data[i, j] = v; X)

# An m×n operand reached as op(P).
semantics_operand(P, op, m, n) = op === identity ? P(m, n) : op(P(n, m))

const SEMANTICS_OPS = (identity, transpose, adjoint)

# Sizes: small (unpacked SIMD fast path), packed path with edge tiles and
# K > 128, K == 0, and empty M / N.
const SEMANTICS_SIZES = [(5, 7, 3), (16, 16, 16), (150, 140, 200), (4, 4, 0), (0, 3, 2), (3, 0, 2)]

@testset "α/β reference-BLAS semantics" begin
    @testset "default kernel, $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        for (M, N, K) in SEMANTICS_SIZES
            check_alpha_beta_semantics(T, M, N, K)
        end
    end
    @testset "$kernel" for (T, kernel) in (
            (Float64,    SIMDKernel{4, 8, 6, Float64}()),
            (Float64,    SIMDKernel{2, 2, 4, Float64}()),
            (Float32,    SIMDKernel{8, 16, 6, Float32}()),
            (ComplexF64, SIMDKernel{4, 4, 6, ComplexF64}()),
            (ComplexF32, SIMDKernel{8, 8, 6, ComplexF32}()),
            (Float64,    ScalarKernel{4, 4}()),
            (Float32,    ScalarKernel{8, 6}()),
            (ComplexF64, ScalarKernel{4, 4}()),
            (ComplexF32, ScalarKernel{8, 6}()),
        )
        for (M, N, K) in SEMANTICS_SIZES
            check_alpha_beta_semantics(T, M, N, K; kernel)
        end
    end
    @testset "no reads, $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        M, N, K = 7, 5, 3
        # α == 0: A and B are never read, through any op(A)/op(B) wrapper.
        for opA in SEMANTICS_OPS, opB in SEMANTICS_OPS, α in (zero(T), false)
            A = semantics_operand((m, n) -> Unreadable{T}(m, n), opA, M, K)
            B = semantics_operand((m, n) -> Unreadable{T}(m, n), opB, K, N)
            for β in (zero(T), one(T), T(-0.5))
                C0 = randn(T, M, N); C = copy(C0)
                gemm!(C, A, B, α, β)
                @test C == T(β) .* C0
            end
        end
        # α == β == 0: C is only written.
        C = WriteOnly(fill(T(NaN), M, N))
        gemm!(C, Unreadable{T}(M, K), Unreadable{T}(K, N), false, false)
        @test all(iszero, C.data)
    end
    @testset "α == 0, β == 1 leaves C untouched, $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        M, N, K = 9, 6, 4
        for opA in SEMANTICS_OPS, opB in SEMANTICS_OPS
            # Non-contiguous poisoned operands (every other row of the parent).
            A = semantics_operand((m, n) -> view(poisoned(T, 2m, n), 1:2:2m, :), opA, M, K)
            B = semantics_operand((m, n) -> view(poisoned(T, 2m, n), 1:2:2m, :), opB, K, N)
            C0 = poisoned(T, M, N); C = copy(C0)
            gemm!(C, A, B, zero(T), one(T))
            @test isequal(C, C0)
        end
    end
    @testset "K == 0 ignores α, $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        for A in (zeros(T, 4, 0), transpose(zeros(T, 0, 4)), adjoint(zeros(T, 0, 4))),
            β in (zero(T), T(-0.5))
            C0 = randn(T, 4, 3); C = copy(C0)
            gemm!(C, A, zeros(T, 0, 3), T(Inf), β)
            @test C == β .* C0
        end
    end
    @testset "dimension checks on early-return paths, $T" for T in (Float64, ComplexF64)
        # α == 0
        @test_throws DimensionMismatch gemm!(zeros(T, 3, 4), Unreadable{T}(3, 2), Unreadable{T}(5, 4), false, true)
        @test_throws DimensionMismatch gemm!(zeros(T, 3, 4), Unreadable{T}(2, 2), Unreadable{T}(2, 4), false, true)
        # empty C
        @test_throws DimensionMismatch gemm!(zeros(T, 0, 4), zeros(T, 1, 2), zeros(T, 2, 4))
        @test_throws DimensionMismatch gemm!(zeros(T, 3, 0), zeros(T, 3, 2), zeros(T, 2, 1))
        # K == 0
        @test_throws DimensionMismatch gemm!(zeros(T, 3, 4), zeros(T, 3, 0), zeros(T, 1, 4))
    end
end
