# gemm.jl — single-threaded, generic, Goto-style matrix multiply with
# selectable, JIT-generated microkernels via multiple dispatch + @generated.
#
# Public API:
#     gemm!(C, A, B, α=true, β=false; kernel=default_kernel(eltype(C)))
#         C ← α·A·B + β·C   (in place)
#
# Available kernels (all parametric):
#     ScalarKernel{MR,NR}        — fully scalar, works for any T
#     SIMDKernel{W,MR,NR,T}      — explicit-width SIMD; W is the lane count
#                                  (LLVM lowers `<W x T>` to zmm/ymm/xmm/...).
#                                  MR must be a multiple of W. Requires
#                                  StridedMatrix C, falls back to
#                                  ScalarKernel{MR,NR} otherwise.
#
# Both kernels' bodies are emitted by `@generated`, so all dimensions are
# fully unrolled by LLVM with no runtime indirection. Try a different shape
# by constructing a different kernel:
#     gemm!(C, A, B; kernel=SIMDKernel{8,16,14,Float64}())   # AVX-512 16×14
#     gemm!(C, A, B; kernel=SIMDKernel{4, 8, 6,Float64}())   # AVX2 8×6

using Base.Cartesian

# ─── SIMD primitives (parametric on lane count W and scalar type T) ──────

const Vec{W,T} = NTuple{W, VecElement{T}}

@inline _vbcast(x::T, ::Val{W}) where {T,W} = ntuple(_ -> VecElement(x), Val(W))
@inline _vzero(::Type{Vec{W,T}}) where {W,T} = _vbcast(zero(T), Val(W))

# Vector load/store via a single Ptr{Vec{W,T}} cast. The kernel body extracts
# raw pointers (and the C column stride) once outside the hot loop and passes
# byte offsets in — that lets LLVM hoist the base computations and keeps the
# inner loop down to one `vmovupd zmm`/`vfmadd231pd` sequence per iteration.
@inline function _vload(::Type{Vec{W,T}}, p::Ptr{T}) where {W,T}
    unsafe_load(Ptr{Vec{W,T}}(p))
end

@inline function _vstore!(p::Ptr{T}, v::Vec{W,T}) where {W,T}
    unsafe_store!(Ptr{Vec{W,T}}(p), v)
    return nothing
end

# Vector FMA, pure Julia. `NTuple{W, VecElement{T}}` is already a `<W x T>`
# at the LLVM level; doing W independent scalar muladds and reassembling via
# `ntuple(_, Val(W))` (fully unrolled) gives LLVM's SLP vectorizer a clean
# pattern to fold back into a single vector FMA. On AVX-512 targets we expect
# `vfmadd231pd zmm,…` (Float64) or `vfmadd231ps zmm,…` (Float32).
@inline _vfma(a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,T} =
    ntuple(Val(W)) do i
        @inbounds VecElement(muladd(a[i].value, b[i].value, c[i].value))
    end

# ─── Kernel selection ──────────────────────────────────────────────────────

abstract type AbstractKernel end

"""
    ScalarKernel{MR,NR}()

Scalar microkernel with MR×NR register-tile accumulator, body unrolled by
`@generated`. Works for any element type. Used as the fallback for SIMD
kernels when C isn't a `StridedMatrix`.
"""
struct ScalarKernel{MR,NR} <: AbstractKernel end

"""
    SIMDKernel{W,MR,NR,T}()

Explicit-width SIMD microkernel. `W` is the SIMD lane count: LLVM lowers
`<W x T>` to whatever width the target supports (zmm for W=8 Float64 on
AVX-512, ymm for W=4 Float64 on AVX2, xmm for W=2 Float64 on SSE2, etc.).
`MR` must be a multiple of `W`; the accumulator is laid out as `MR/W` row
groups × `NR` columns of vector registers. Requires `StridedMatrix{T}` for
the SIMD path; falls back to `ScalarKernel{MR,NR}` otherwise.

Examples (Float64):
    SIMDKernel{8,  8, 24, Float64}()   # AVX-512, MKL-style 8×24 (default)
    SIMDKernel{8, 16, 14, Float64}()   # AVX-512, BLIS skx-style 16×14
    SIMDKernel{4,  4,  6, Float64}()   # AVX2 ymm, 4×6
    SIMDKernel{4,  8,  6, Float64}()   # AVX2 ymm, 8×6 (2 ymm per col)
    SIMDKernel{2,  2,  4, Float64}()   # SSE2 xmm, 2×4
"""
struct SIMDKernel{W,MR,NR,T} <: AbstractKernel end

mr(::ScalarKernel{MR,NR}) where {MR,NR}             = MR
nr(::ScalarKernel{MR,NR}) where {MR,NR}             = NR
mr(::SIMDKernel{W,MR,NR,T}) where {W,MR,NR,T}       = MR
nr(::SIMDKernel{W,MR,NR,T}) where {W,MR,NR,T}       = NR

default_kernel(::Type{Float64}) = SIMDKernel{8,   8, 24, Float64}()
default_kernel(::Type{Float32}) = SIMDKernel{16, 16, 24, Float32}()
default_kernel(::Type{T}) where {T} = ScalarKernel{8,6}()

# ─── Block sizes (per-kernel traits) ──────────────────────────────────────
#
# `mc_block`: rows of A panel held in L2. Must be a multiple of MR so the
#             macrokernel's ir-loop has no edge panels in the common case.
# `kc_block`: depth of A/B panels. For SIMD kernels we cap KC so the B
#             micropanel (NR × KC × sizeof(T)) fits in L1d alongside the A
#             micropanel; ~28 KiB target leaves headroom in a 32 KiB L1.
# `nc_block`: width of B slab held in L3. 4080 is divisible by 6, 14, 24.
#
# Override these for new kernel shapes if you care about boundary edges or
# L1 capacity on a specific microarch.

mc_block(k::AbstractKernel) = cld(72, mr(k)) * mr(k)   # round up to multiple of MR
kc_block(::AbstractKernel)  = 256
nc_block(::AbstractKernel)  = 4080

# SIMD kernels: scale MC so the A micropanel byte budget (~MC × KC × sizeof(T))
# stays L2-resident. Reference point: Float64 MR=8 → MC=72 (BLIS dgemm skx default).
# For Float32 the same byte budget gives MC=144; matches BLIS sgemm skx default.
function mc_block(::SIMDKernel{W,MR,NR,T}) where {W,MR,NR,T}
    target_mc = (72 * 8) ÷ sizeof(T)
    return cld(target_mc, MR) * MR
end

# SIMD kernels: cap KC so the B micropanel (NR × KC × sizeof(T)) fits in L1.
function kc_block(::SIMDKernel{W,MR,NR,T}) where {W,MR,NR,T}
    raw = (28 * 1024) ÷ (NR * sizeof(T))
    return clamp(raw - raw % 8, 64, 256)
end

# ─── Main entry point ─────────────────────────────────────────────────────

function gemm!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T},
               α = true, β = false;
               kernel::AbstractKernel = default_kernel(T)) where {T}
    M, N = size(C)
    size(A, 1) == M           || throw(DimensionMismatch("size(A,1) ≠ size(C,1)"))
    size(B, 2) == N           || throw(DimensionMismatch("size(B,2) ≠ size(C,2)"))
    size(A, 2) == size(B, 1)  || throw(DimensionMismatch("size(A,2) ≠ size(B,1)"))
    K = size(A, 2)

    _scale!(C, β)
    (M == 0 || N == 0 || K == 0) && return C

    MR_ = mr(kernel)
    NR_ = nr(kernel)
    MC_ = mc_block(kernel)
    KC_ = kc_block(kernel)
    NC_ = nc_block(kernel)

    mc_max = min(MC_, M)
    kc_max = min(KC_, K)
    nc_max = min(NC_, N)
    Apack = Vector{T}(undef, cld(mc_max, MR_) * MR_ * kc_max)
    Bpack = Vector{T}(undef, cld(nc_max, NR_) * NR_ * kc_max)
    αT    = convert(T, α)

    @inbounds for jc in 1:NC_:N
        nc = min(NC_, N - jc + 1)
        for pc in 1:KC_:K
            kc = min(KC_, K - pc + 1)
            _pack_B!(Bpack, B, pc, jc, kc, nc, Val(NR_))
            for ic in 1:MC_:M
                mc = min(MC_, M - ic + 1)
                _pack_A!(Apack, A, ic, pc, mc, kc, Val(MR_))
                _macrokernel!(kernel, C, Apack, Bpack, ic, jc, mc, nc, kc, αT)
            end
        end
    end
    return C
end

function _scale!(C::AbstractMatrix{T}, β) where {T}
    if iszero(β)
        fill!(C, zero(T))
    elseif !isone(β)
        βT = convert(T, β)
        @inbounds @simd for i in eachindex(C)
            C[i] = βT * C[i]
        end
    end
    return C
end

# ─── Packing ──────────────────────────────────────────────────────────────

function _pack_A!(Apack::Vector{T}, A::AbstractMatrix{T},
                  ic::Int, pc::Int, mc::Int, kc::Int, ::Val{MR}) where {T,MR}
    z = zero(T)
    npanels = cld(mc, MR)
    @inbounds for p in 0:npanels-1
        rs   = p * MR
        rmax = min(MR, mc - rs)
        for k in 0:kc-1
            base = p * MR * kc + k * MR
            for i in 0:rmax-1
                Apack[base + i + 1] = A[ic + rs + i, pc + k]
            end
            for i in rmax:MR-1
                Apack[base + i + 1] = z
            end
        end
    end
    return Apack
end

function _pack_B!(Bpack::Vector{T}, B::AbstractMatrix{T},
                  pc::Int, jc::Int, kc::Int, nc::Int, ::Val{NR}) where {T,NR}
    z = zero(T)
    npanels = cld(nc, NR)
    @inbounds for p in 0:npanels-1
        cs   = p * NR
        cmax = min(NR, nc - cs)
        for k in 0:kc-1
            base = p * NR * kc + k * NR
            for j in 0:cmax-1
                Bpack[base + j + 1] = B[pc + k, jc + cs + j]
            end
            for j in cmax:NR-1
                Bpack[base + j + 1] = z
            end
        end
    end
    return Bpack
end

# ─── Macrokernel ──────────────────────────────────────────────────────────

function _macrokernel!(kernel::AbstractKernel, C::AbstractMatrix{T},
                       Apack::Vector{T}, Bpack::Vector{T},
                       ic::Int, jc::Int, mc::Int, nc::Int, kc::Int, α::T) where {T}
    MR_ = mr(kernel)
    NR_ = nr(kernel)
    @inbounds for jr in 0:NR_:nc-1
        nr_ = min(NR_, nc - jr)
        bo = (jr ÷ NR_) * NR_ * kc
        for ir in 0:MR_:mc-1
            mr_ = min(MR_, mc - ir)
            ao = (ir ÷ MR_) * MR_ * kc
            ci = ic + ir
            cj = jc + jr
            if mr_ == MR_ && nr_ == NR_
                _kernel!(kernel, C, Apack, Bpack, ao, bo, kc, ci, cj, MR_, NR_, α, Val(true))
            else
                _kernel!(kernel, C, Apack, Bpack, ao, bo, kc, ci, cj, mr_, nr_, α, Val(false))
            end
        end
    end
    return nothing
end

# ─── Microkernel: ScalarKernel (parametric, JIT-unrolled) ────────────────

@generated function _kernel!(::ScalarKernel{MR,NR}, C::AbstractMatrix{T},
                              Apack::Vector{T}, Bpack::Vector{T},
                              ao::Int, bo::Int, kc::Int, ci::Int, cj::Int,
                              mr_::Int, nr_::Int, α::T,
                              ::Val{full}) where {MR,NR,T,full}
    init      = Expr[]
    inner     = Expr[]
    write_full = Expr[]
    write_edge = Expr[]

    for j in 1:NR, i in 1:MR
        push!(init, :( $(Symbol("c_", i, "_", j)) = zero($T) ))
    end
    for i in 1:MR
        push!(inner, :( $(Symbol("a_", i)) = Apack[ak + $i] ))
    end
    for j in 1:NR
        push!(inner, :( $(Symbol("b_", j)) = Bpack[bk + $j] ))
        for i in 1:MR
            c = Symbol("c_", i, "_", j); a = Symbol("a_", i); b = Symbol("b_", j)
            push!(inner, :( $c = muladd($a, $b, $c) ))
        end
    end
    for j in 1:NR, i in 1:MR
        c = Symbol("c_", i, "_", j)
        push!(write_full, :( C[ci + $(i-1), cj + $(j-1)] =
                             muladd(α, $c, C[ci + $(i-1), cj + $(j-1)]) ))
        push!(write_edge, :( if $i <= mr_ && $j <= nr_
            C[ci + $(i-1), cj + $(j-1)] =
                muladd(α, $c, C[ci + $(i-1), cj + $(j-1)])
        end ))
    end

    quote
        $(Expr(:meta, :inline))
        $(init...)
        @inbounds for k in 0:kc-1
            ak = ao + k * $MR
            bk = bo + k * $NR
            $(inner...)
        end
        if $full
            @inbounds begin $(write_full...) end
        else
            @inbounds begin $(write_edge...) end
        end
        return nothing
    end
end

# ─── Microkernel: SIMDKernel (parametric on W, MR, NR, T; JIT-unrolled) ──

# Non-strided fallback: route through the scalar kernel.
@inline function _kernel!(::SIMDKernel{W,MR,NR,T}, C::AbstractMatrix{T},
                          Apack::Vector{T}, Bpack::Vector{T},
                          ao::Int, bo::Int, kc::Int, ci::Int, cj::Int,
                          mr_::Int, nr_::Int, α::T, v::Val) where {W,MR,NR,T}
    return _kernel!(ScalarKernel{MR,NR}(), C, Apack, Bpack, ao, bo, kc,
                    ci, cj, mr_, nr_, α, v)
end

@generated function _kernel!(::SIMDKernel{W,MR,NR,T}, C::StridedMatrix{T},
                              Apack::Vector{T}, Bpack::Vector{T},
                              ao::Int, bo::Int, kc::Int, ci::Int, cj::Int,
                              mr_::Int, nr_::Int, α::T,
                              ::Val{full}) where {W,MR,NR,T,full}
    MR % W == 0 ||
        throw(ArgumentError("SIMDKernel: MR=$MR must be a multiple of W=$W"))
    rows = MR ÷ W
    sz   = sizeof(T)
    Vty  = :(Vec{$W,$T})

    # Accumulator init
    init = Expr[]
    for r in 1:rows, j in 1:NR
        push!(init, :( $(Symbol("c_", r, "_", j)) = _vzero($Vty) ))
    end

    # Inner k-loop body. `pA`/`pB` are extracted once at the top of the kernel
    # so the loop sees byte-offset arithmetic only.
    inner = Expr[]
    for r in 1:rows
        a   = Symbol("a_", r)
        off = (r - 1) * W
        push!(inner, :( $a = _vload($Vty, pA + (ao + k * $MR + $off) * $sz) ))
    end
    for j in 1:NR
        bv = Symbol("bv_", j)
        push!(inner, :( $bv = _vbcast(unsafe_load(pB, bo + k * $NR + $j), Val($W)) ))
        for r in 1:rows
            c = Symbol("c_", r, "_", j); a = Symbol("a_", r)
            push!(inner, :( $c = _vfma($a, $bv, $c) ))
        end
    end

    # Vector writeback (full tile). `pC`/`ldc` extracted once outside.
    write_full = Expr[]
    for j in 1:NR, r in 1:rows
        c    = Symbol("c_", r, "_", j)
        roff = (r - 1) * W
        push!(write_full, quote
            let off = ((cj + $(j-1) - 1) * ldc + (ci + $roff - 1)) * $sz
                _vstore!(pC + off, _vfma(αv, $c, _vload($Vty, pC + off)))
            end
        end)
    end

    # Scalar edge writeback
    write_edge = Expr[]
    for j in 1:NR, r in 1:rows, ii in 1:W
        c  = Symbol("c_", r, "_", j)
        gi = (r - 1) * W + ii          # 1-based row in the MR×NR tile
        push!(write_edge, quote
            if $gi <= mr_ && $j <= nr_
                cv = $c
                C[ci + $(gi-1), cj + $(j-1)] =
                    muladd(α, cv[$ii].value, C[ci + $(gi-1), cj + $(j-1)])
            end
        end)
    end

    quote
        $(Expr(:meta, :inline))
        $(init...)
        GC.@preserve Apack Bpack C begin
            pA = pointer(Apack)
            pB = pointer(Bpack)

            @inbounds for k in 0:kc-1
                $(inner...)
            end

            pC  = pointer(C)
            ldc = stride(C, 2)
            if $full
                αv = _vbcast(α, Val($W))
                $(write_full...)
            else
                @inbounds begin $(write_edge...) end
            end
        end
        return nothing
    end
end
