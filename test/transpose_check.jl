# Correctness sweep: `gemm!` over every combination of op(A), op(B) ∈
# {N, T, C} and every supported eltype. Compares result against Julia's
# `op(A) * op(B)` reference.
#
# Run from the package root:
#     julia --project=. test/transpose_check.jl
#
# Perf comparisons against OpenBLAS / MKL live in
# `bench/transpose_bench_{real,complex}.jl`.

using JuBLAS, LinearAlgebra, Random, Test
using LinearAlgebra: Symmetric, Hermitian

Random.seed!(0)

apply_op(A, op::Symbol) = op === :N ? A :
                          op === :T ? transpose(A) :
                          adjoint(A)

# Build a matrix of shape (rows, cols) reachable as `apply_op(parent, op)`.
# Returns `(parent_storage, view_used_in_gemm)`. Parent must stay rooted
# because `transpose`/`adjoint` are lazy wrappers.
function make_arg(::Type{T}, op::Symbol, rows::Int, cols::Int) where {T}
    if op === :N
        P = randn(T, rows, cols)
        return P, P
    else
        P = randn(T, cols, rows)
        return P, apply_op(P, op)
    end
end

# Block-aligned + edge sizes. Edges hit the masked-tail path; aligned go
# through the bulk pointer-load path.
const SIZES = [(32, 32, 32), (50, 50, 50), (37, 41, 23),
               (64, 64, 64), (128, 96, 80), (73, 89, 127), (256, 256, 256)]

@testset "gemm! transpose/adjoint" begin
    @testset "$T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol = 1e3 * eps(real(T))
        @testset "size=$((M,N,K)) opA=$opA opB=$opB" for (M, N, K) in SIZES,
                                                         opA in (:N, :T, :C),
                                                         opB in (:N, :T, :C)
            _, A = make_arg(T, opA, M, K)
            _, B = make_arg(T, opB, K, N)
            C0 = randn(T, M, N)
            C  = copy(C0)
            α  = T(2.5); β = T(-1.5)
            ref = α .* (A * B) .+ β .* C0
            gemm!(C, A, B, α, β)
            err = maximum(abs, C .- ref) / max(eps(real(T)), maximum(abs, ref))
            @test err < tol
        end
    end
end

# Builds a `Symmetric` / `Hermitian` view from a fresh random parent. The
# parent's unstored triangle is left as junk to confirm `gemm!` only
# reads the stored side (via `Matrix(::Symmetric/Hermitian)`).
function make_wrapped(::Type{T}, kind::Symbol, n::Int) where {T}
    P = randn(T, n, n)
    return kind === :SymU  ? Symmetric(P, :U) :
           kind === :SymL  ? Symmetric(P, :L) :
           kind === :HermU ? Hermitian(P, :U) :
           kind === :HermL ? Hermitian(P, :L) :
           error("unknown wrap kind $kind")
end

@testset "gemm! Symmetric/Hermitian" begin
    @testset "$T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol = 1e3 * eps(real(T))
        # `Hermitian` over real eltypes is identical to `Symmetric`, so
        # only test it for complex where it does conjugate-mirroring.
        kinds = T <: Complex ? (:SymU, :SymL, :HermU, :HermL) : (:SymU, :SymL)
        N = 64

        @testset "A=$kind, B=Matrix" for kind in kinds
            A = make_wrapped(T, kind, N)
            B = randn(T, N, N)
            C0 = randn(T, N, N)
            C  = copy(C0)
            α  = T(2.5); β = T(-1.5)
            ref = α .* (Matrix(A) * B) .+ β .* C0
            gemm!(C, A, B, α, β)
            @test maximum(abs, C .- ref) / max(eps(real(T)), maximum(abs, ref)) < tol
        end

        @testset "A=Matrix, B=$kind" for kind in kinds
            A = randn(T, N, N)
            B = make_wrapped(T, kind, N)
            C0 = randn(T, N, N)
            C  = copy(C0)
            α  = T(2.5); β = T(-1.5)
            ref = α .* (A * Matrix(B)) .+ β .* C0
            gemm!(C, A, B, α, β)
            @test maximum(abs, C .- ref) / max(eps(real(T)), maximum(abs, ref)) < tol
        end

        @testset "A=$kindA, B=$kindB" for kindA in kinds, kindB in kinds
            A = make_wrapped(T, kindA, N)
            B = make_wrapped(T, kindB, N)
            C0 = randn(T, N, N)
            C  = copy(C0)
            α  = T(2.5); β = T(-1.5)
            ref = α .* (Matrix(A) * Matrix(B)) .+ β .* C0
            gemm!(C, A, B, α, β)
            @test maximum(abs, C .- ref) / max(eps(real(T)), maximum(abs, ref)) < tol
        end
    end
end
