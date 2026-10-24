# ScalarKernel gemm (real and complex) against a higher-precision reference:
# every op(A)/op(B) ∈ {N, T, C} combination, strided views whose column
# stride exceeds their row count, edge tiles, multiple KC/MC/NC blocks, a
# sweep of α/β values, and for complex also stepped/reversed/indexed views
# and Complex{BigFloat}.

using JuBLAS, LinearAlgebra, Random, Test

Random.seed!(3)

const SK_SIZES   = (0, 1, 3, 4, 5, 9, 17, 33)
const SK_KERNELS = (ScalarKernel{4, 4}(), ScalarKernel{8, 6}())
const SK_OPS     = (:N, :T, :C)

sk_alphas(::Type{T}) where {T} =
    T <: Complex ? (zero(T), one(T), T(2.5), T(0.75, -1.25)) : (zero(T), one(T), T(2.5), T(-1.25))
sk_betas(::Type{T}) where {T} =
    T <: Complex ? (zero(T), one(T), T(-0.5), T(-0.25, 1.5)) : (zero(T), one(T), T(-0.5), T(3))

# Reference eltype: Float64 for the single-precision types, BigFloat for
# the double-precision ones.
sk_reftype(::Type{Float32})    = Float64
sk_reftype(::Type{ComplexF32}) = ComplexF64
sk_reftype(::Type{Float64})    = BigFloat
sk_reftype(::Type{ComplexF64}) = Complex{BigFloat}
sk_reftype(::Type{T}) where {T<:Union{BigFloat,Complex{BigFloat}}} = T

sk_apply(op, X) = op === :N ? X : op === :T ? transpose(X) : adjoint(X)

# An m×n operand reached as op(X), where X is a view into a larger matrix
# (column stride > rows) unless `asview == false`.
function sk_operand(::Type{T}, op, m, n; asview = true) where {T}
    r, c = op === :N ? (m, n) : (n, m)
    X = asview ? view(randn(T, r + 4, c + 3), 2:r+1, 3:c+2) : randn(T, r, c)
    return sk_apply(op, X)
end

# Checks `C` against `α·AB + β·C0` elementwise with the standard gemm
# forward-error bound γ·(|α|·|A||B| + |β|·|C0|).
# Same, but X is a stepped, reversed or index-vector view (rows only).
function sk_layout_operand(::Type{T}, op, layout, m, n) where {T}
    r, c = op === :N ? (m, n) : (n, m)
    X = layout === :stepped  ? view(randn(T, 2r + 1, c), 2:2:2r, :) :
        layout === :reversed ? view(randn(T, r, c), r:-1:1, :) :
                               view(randn(T, r + 3, c), randperm(r + 3)[1:r], :)
    return sk_apply(op, X)
end

function sk_check(C, ABref, Aabs, Babs, C0, α, β, K)
    T  = eltype(C)
    RT = sk_reftype(T)
    ref   = RT(α) .* ABref .+ RT(β) .* RT.(C0)
    bound = abs(α) .* (Aabs * Babs) .+ abs(β) .* abs.(C0)
    γ     = 8 * (K + 4) * eps(real(T))
    return all(abs.(RT.(C) .- ref) .<= γ .* bound .+ floatmin(real(T)))
end

# Runs one gemm on a view of a larger C and checks both the result and
# that nothing outside the view was touched.
function sk_run(kernel, A, B, α, β, ABref, Aabs, Babs; Cview = true)
    T = eltype(A)
    M, K = size(A); N = size(B, 2)
    Cp  = Cview ? randn(T, M + 5, N + 3) : randn(T, M, N)
    Cp0 = copy(Cp)
    C   = Cview ? view(Cp, 3:M+2, 2:N+1) : Cp
    C0  = copy(C)
    JuBLAS._gemm!(C, A, B, α, β, kernel, nothing, nothing)
    ok = sk_check(C, ABref, Aabs, Babs, C0, α, β, K)
    if Cview
        mask = trues(size(Cp)); mask[3:M+2, 2:N+1] .= false
        ok &= Cp[mask] == Cp0[mask]
    end
    return ok
end

function sk_refs(A, B)
    RT = sk_reftype(eltype(A))
    return RT.(Matrix(A)) * RT.(Matrix(B)), abs.(Matrix(A)), abs.(Matrix(B))
end

@testset "ScalarKernel gemm" begin
    setprecision(BigFloat, 128) do
        @testset "$T size sweep (views)" for T in (Float32, Float64, ComplexF32, ComplexF64)
            αs, βs = sk_alphas(T), sk_betas(T)
            idx = 0
            for m in SK_SIZES, n in SK_SIZES, k in SK_SIZES, opA in SK_OPS, opB in SK_OPS
                A = sk_operand(T, opA, m, k)
                B = sk_operand(T, opB, k, n)
                refs = sk_refs(A, B)
                for kernel in SK_KERNELS
                    # Rotate through all 16 (α, β) pairs across cases.
                    idx += 1
                    α = αs[mod1(idx, 4)]; β = βs[mod1(cld(idx, 4), 4)]
                    @test sk_run(kernel, A, B, α, β, refs...)
                end
            end
        end

        @testset "$T all α/β, Matrix and view operands" for T in (Float32, Float64, ComplexF32, ComplexF64)
            for (m, n, k) in ((5, 9, 3), (17, 33, 9)),
                opA in SK_OPS, opB in SK_OPS, asview in (false, true)
                A = sk_operand(T, opA, m, k; asview)
                B = sk_operand(T, opB, k, n; asview)
                refs = sk_refs(A, B)
                # Plain-Matrix operands only for the {4,4} shape (compile time).
                kernels = asview ? SK_KERNELS : (ScalarKernel{4, 4}(),)
                for kernel in kernels, α in sk_alphas(T), β in sk_betas(T)
                    @test sk_run(kernel, A, B, α, β, refs...; Cview = asview)
                end
            end
        end

        # K > KC (= 256) and M > MC (= 72) for both kernel shapes.
        @testset "$T multi-block" for T in (Float32, Float64, ComplexF32, ComplexF64)
            m, n, k = 80, 13, 300
            for opA in SK_OPS, opB in SK_OPS
                A = sk_operand(T, opA, m, k)
                B = sk_operand(T, opB, k, n)
                refs = sk_refs(A, B)
                for kernel in SK_KERNELS
                    @test sk_run(kernel, A, B, last(sk_alphas(T)), last(sk_betas(T)), refs...)
                end
            end
        end

        # N > NC (= 4080): several jc blocks.
        @testset "$T multiple NC blocks" for T in (Float64, ComplexF64)
            m, n, k = 5, 4100, 7
            for opA in SK_OPS, opB in SK_OPS
                A = sk_operand(T, opA, m, k)
                B = sk_operand(T, opB, k, n)
                @test sk_run(ScalarKernel{4, 4}(), A, B, last(sk_alphas(T)), last(sk_betas(T)), sk_refs(A, B)...)
            end
        end

        # Complex packing copies element by element, so any AbstractMatrix
        # works as A or B.
        @testset "$T stepped/reversed/indexed operands" for T in (ComplexF32, ComplexF64)
            m, n, k = 17, 9, 33
            for layout in (:stepped, :reversed, :indexed), opA in SK_OPS, opB in SK_OPS
                A = sk_layout_operand(T, opA, layout, m, k)
                B = sk_layout_operand(T, opB, layout, k, n)
                refs = sk_refs(A, B)
                for kernel in SK_KERNELS
                    @test sk_run(kernel, A, B, last(sk_alphas(T)), last(sk_betas(T)), refs...)
                end
            end
        end

        @testset "Complex{BigFloat}" begin
            T = Complex{BigFloat}
            for opA in SK_OPS, opB in SK_OPS
                A = sk_operand(T, opA, 9, 7)
                B = sk_operand(T, opB, 7, 5)
                refs = sk_refs(A, B)
                for kernel in SK_KERNELS
                    @test sk_run(kernel, A, B, last(sk_alphas(T)), last(sk_betas(T)), refs...)
                end
            end
            # Default kernel through the public API.
            A = randn(T, 9, 7); B = randn(T, 7, 5); C0 = randn(T, 9, 5)
            C = copy(C0)
            gemm!(C, A, B, T(0.75, -1.25), T(-0.25, 1.5))
            @test C ≈ T(0.75, -1.25) .* (A * B) .+ T(-0.25, 1.5) .* C0 rtol = 1e-30
        end
    end

    # Explicit ScalarKernels never need the CPU's cache sizes or SIMD width.
    @testset "no CPU detection" begin
        saved = (JuBLAS._CACHE_SIZES_CACHE[], JuBLAS._SIMD_BYTES_CACHE[])
        try
            JuBLAS._CACHE_SIZES_CACHE[] = (0, 0, 0); JuBLAS._SIMD_BYTES_CACHE[] = -1
            for T in (Float64, ComplexF64)
                kernel = ScalarKernel{4, 4}()
                A = randn(T, 80, 300); B = randn(T, 300, 13); C = zeros(T, 80, 13)
                gemm!(C, A, B; kernel)
                @test C ≈ A * B
                Apack, Bpack = gemm_workspace(T, kernel)
                gemm!(C, A, B; kernel, Apack, Bpack)
                @test C ≈ A * B
            end
            @test JuBLAS._CACHE_SIZES_CACHE[] == (0, 0, 0)
            @test JuBLAS._SIMD_BYTES_CACHE[] == -1
        finally
            JuBLAS._CACHE_SIZES_CACHE[], JuBLAS._SIMD_BYTES_CACHE[] = saved
        end
    end

    @testset "complex public API" begin
        for T in (ComplexF32, ComplexF64)
            A = randn(T, 37, 29); B = randn(T, 29, 41); C0 = randn(T, 37, 41)
            α = T(1.5, -0.5); β = T(0.25, 2.0)
            ref = α .* (A * B) .+ β .* C0
            rtol = 100 * eps(real(T))
            # ScalarKernel through the kwarg `gemm!`, with and without workspace.
            kernel = ScalarKernel{4, 4}()
            C = copy(C0)
            gemm!(C, A, B, α, β; kernel)
            @test C ≈ ref rtol=rtol
            Apack, Bpack = gemm_workspace(T, kernel)
            @test Apack isa Vector{T} && Bpack isa Vector{T}
            C = copy(C0)
            At = copy(A'); Bt = copy(transpose(B))
            gemm!(C, At', transpose(Bt), α, β; kernel, Apack, Bpack)
            @test C ≈ ref rtol=rtol
            # Non-strided C with a complex SIMD kernel routes to ScalarKernel.
            rows = [1, 3, 4, 7, 8, 10, 12, 15, 17, 18, 20, 23, 25, 26, 28, 31, 33, 34, 36, 39,
                    40, 41, 43, 45, 46, 48, 50, 52, 53, 55, 57, 58, 60, 62, 63, 65, 67]
            Cp = randn(T, 70, 41); Cp0 = copy(Cp)
            Cv = view(Cp, rows, :)
            @test !(Cv isa StridedMatrix)
            for kw in ((; kernel = T === ComplexF64 ? SIMDKernel{4, 4, 6, ComplexF64}() :
                                                      SIMDKernel{8, 8, 6, ComplexF32}()), (;))
                Cp .= Cp0
                gemm!(Cv, A, B, α, β; kw...)
                @test Cv ≈ α .* (A * B) .+ β .* Cp0[rows, :] rtol=rtol
                mask = trues(size(Cp)); mask[rows, :] .= false
                @test Cp[mask] == Cp0[mask]
            end
        end
    end
end
