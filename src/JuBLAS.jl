module JuBLAS

using CpuId
using SIMD
using LinearAlgebra: Transpose, Adjoint, Symmetric, Hermitian

include("utils.jl")
include("gemm.jl")
include("gemm_complex.jl")

# `Symmetric` / `Hermitian` inputs: materialize the wrapped matrix to a
# regular `Matrix` and re-enter `gemm!`; the eltype-dispatched real or
# complex `gemm!` then handles it normally. `Matrix(::Symmetric)` mirrors
# the stored triangle without conjugation; `Matrix(::Hermitian{<:Complex})`
# conjugate-mirrors. The N×N temporary is the cost of this simplicity —
# a real `symm!` / `hemm!` kernel could read the stored triangle in place
# (still on the TODO list).
#
# Each forward is split into a `T<:Real` and a `T<:Complex{TR<:Real}`
# version mirroring the existing `gemm!` eltype constraints, so each
# new method is a strict refinement of the corresponding existing one.
# A single `where {T}` version was tried first — it ambiguates because
# the wrapper constraint and the existing eltype constraint are
# incomparable (mine accepts T outside Real ∪ Complex; existing complex
# rejects T outside that set).
const _SymHerm{T} = Union{Symmetric{T}, Hermitian{T}}

# Real eltype.
@inline gemm!(C::AbstractMatrix{T}, A::_SymHerm{T}, B::AbstractMatrix{T},
              α = true, β = false; kwargs...) where {T<:Real} =
    gemm!(C, Matrix(A), B, α, β; kwargs...)
@inline gemm!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::_SymHerm{T},
              α = true, β = false; kwargs...) where {T<:Real} =
    gemm!(C, A, Matrix(B), α, β; kwargs...)
@inline gemm!(C::AbstractMatrix{T}, A::_SymHerm{T}, B::_SymHerm{T},
              α = true, β = false; kwargs...) where {T<:Real} =
    gemm!(C, Matrix(A), Matrix(B), α, β; kwargs...)

# Complex eltype.
@inline gemm!(C::AbstractMatrix{Complex{TR}}, A::_SymHerm{Complex{TR}}, B::AbstractMatrix{Complex{TR}},
              α = true, β = false; kwargs...) where {TR<:Real} =
    gemm!(C, Matrix(A), B, α, β; kwargs...)
@inline gemm!(C::AbstractMatrix{Complex{TR}}, A::AbstractMatrix{Complex{TR}}, B::_SymHerm{Complex{TR}},
              α = true, β = false; kwargs...) where {TR<:Real} =
    gemm!(C, A, Matrix(B), α, β; kwargs...)
@inline gemm!(C::AbstractMatrix{Complex{TR}}, A::_SymHerm{Complex{TR}}, B::_SymHerm{Complex{TR}},
              α = true, β = false; kwargs...) where {TR<:Real} =
    gemm!(C, Matrix(A), Matrix(B), α, β; kwargs...)

export gemm!, gemm_workspace, default_kernel,
       AbstractKernel, ScalarKernel, SIMDKernel
end
