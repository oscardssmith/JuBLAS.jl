# gemm! on operands the pointer-based pack loops and SIMD writeback can't
# address directly: stepped and reversed rows, index-vector views, and
# transposes/adjoints of those. Checks the result and that nothing outside
# C's view (or anywhere in A's and B's parents) is written.

using JuBLAS, LinearAlgebra, Random, Test

Random.seed!(4)

# r×c matrix with the given layout, returned as (view, parent).
function layout_matrix(::Type{T}, layout, r, c) where {T}
    if layout === :dense
        P = randn(T, r, c);         return P, P
    elseif layout === :padded       # unit row stride, column stride > r
        P = randn(T, r + 3, c + 2); return view(P, 2:r+1, 2:c+1), P
    elseif layout === :stepped
        P = randn(T, 2r + 1, c);    return view(P, 2:2:2r, :), P
    elseif layout === :reversed
        P = randn(T, r, c + 1);     return view(P, r:-1:1, 2:c+1), P
    else # :indexed (not strided)
        P = randn(T, r + 3, c);     return view(P, randperm(r + 3)[1:r], :), P
    end
end

# An m×n operand reached as op(X), X with the given layout.
function layout_operand(::Type{T}, op, layout, m, n) where {T}
    X, P = op === :N ? layout_matrix(T, layout, m, n) : layout_matrix(T, layout, n, m)
    return (op === :N ? X : op === :T ? transpose(X) : adjoint(X)), P
end

const LAYOUTS   = (:dense, :padded, :stepped, :reversed, :indexed)
const C_LAYOUTS = (:dense, :padded, :stepped, :reversed, :indexed)
const OPS       = (:N, :T, :C)

function check_layout_gemm(::Type{T}, (M, N, K), (opA, lA), (opB, lB), lC; kw...) where {T}
    A, PA = layout_operand(T, opA, lA, M, K); PA0 = copy(PA)
    B, PB = layout_operand(T, opB, lB, K, N); PB0 = copy(PB)
    C, PC = layout_matrix(T, lC, M, N)
    C0 = copy(C)
    # Mark C's elements in the parent so we can check the rest is untouched.
    outside = trues(size(PC)); fill!(view(outside, parentindices(C)...), false)
    lC === :dense && (outside .= false)
    PC0 = copy(PC)
    α, β = T(1.5), T(-0.5)
    ref = α .* (Matrix(A) * Matrix(B)) .+ β .* C0
    gemm!(C, A, B, α, β; kw...)
    tol = 100 * K * eps(real(T))
    return isapprox(C, ref; rtol = tol) && PC[outside] == PC0[outside] &&
           PA == PA0 && PB == PB0
end

# (37, 29, 41): edge tiles, packed path. (90, 70, 600): K > KC, so
# several pack/kernel rounds accumulate into C.
const LAYOUT_SIZES = ((37, 29, 41), (90, 70, 600))

@testset "gemm! operand layouts" begin
    @testset "$T" for T in (Float32, Float64, ComplexF32, ComplexF64)
        # Non-strided complex C isn't supported by the complex SIMD kernels.
        c_layouts = T <: Complex ? filter(!=(:indexed), C_LAYOUTS) : C_LAYOUTS
        for sz in LAYOUT_SIZES
            for opA in OPS, lA in LAYOUTS
                @test check_layout_gemm(T, sz, (opA, lA), (:N, :dense), :dense)
            end
            for opB in OPS, lB in LAYOUTS
                @test check_layout_gemm(T, sz, (:N, :dense), (opB, lB), :dense)
            end
            for lC in c_layouts
                @test check_layout_gemm(T, sz, (:N, :dense), (:N, :dense), lC)
                @test check_layout_gemm(T, sz, (:C, :stepped), (:T, :reversed), lC)
            end
        end
        T <: Real || continue
        # Real ScalarKernel shares the packing code.
        for opA in OPS, lA in LAYOUTS, lC in C_LAYOUTS
            @test check_layout_gemm(T, (37, 29, 41), (opA, lA), (:T, :stepped), lC;
                                    kernel = ScalarKernel{4, 4}())
        end
    end
end
