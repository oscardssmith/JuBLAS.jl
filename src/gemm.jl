# gemm.jl — single-threaded, generic, Goto-style matrix multiply with
# selectable, JIT-generated microkernels via multiple dispatch + @generated.
#
# Public API:
#     gemm!(C, A, B, α=true, β=false; kernel=default_kernel(eltype(C)))
#         C ← α·A·B + β·C   (in place)
#
# Available kernels (all parametric):
#     ScalarKernel{MR,NR}        — fully scalar, works for any T
#     AVX512Kernel{MR,NR,T}      — explicit zmm SIMD; T ∈ {Float32, Float64};
#                                  MR must be a multiple of the SIMD lane count
#                                  (W=8 for Float64, W=16 for Float32);
#                                  requires StridedMatrix C, falls back to
#                                  ScalarKernel{MR,NR} otherwise.
#
# Both kernels' bodies are emitted by `@generated`, so `(MR,NR)` are
# fully unrolled by LLVM with no runtime indirection. Try a different shape
# by constructing a different kernel:
#     gemm!(C, A, B; kernel=AVX512Kernel{16,14,Float64}())

using Base.Cartesian

# ─── SIMD primitives (parametric on lane count W and scalar type T) ──────

const Vec{W,T} = NTuple{W, VecElement{T}}

@inline _vbcast(x::T, ::Val{W}) where {T,W} = ntuple(_ -> VecElement(x), Val(W))
@inline _vzero(::Type{Vec{W,T}}) where {W,T} = _vbcast(zero(T), Val(W))
@inline _vload(::Type{Vec{W,T}}, p::Ptr{T}) where {W,T} = unsafe_load(Ptr{Vec{W,T}}(p))
@inline _vstore!(p::Ptr{T}, v::Vec{W,T}) where {W,T} = unsafe_store!(Ptr{Vec{W,T}}(p), v)

# Vector FMA via llvmcall (full-module form). Emits `vfmadd231pd zmm,…` (Float64)
# or `vfmadd231ps zmm,…` (Float32) when the target supports AVX-512.
@generated function _vfma(a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,T}
    bits = sizeof(T) * 8
    elt  = bits == 64 ? "double" : (bits == 32 ? "float" : error("unsupported eltype"))
    vty  = "<$W x $elt>"
    fn   = "llvm.fma.v$(W)f$(bits)"
    ir   = """
        declare $vty @$fn($vty, $vty, $vty)

        define $vty @entry($vty, $vty, $vty) #0 {
        top:
            %r = call $vty @$fn($vty %0, $vty %1, $vty %2)
            ret $vty %r
        }

        attributes #0 = { alwaysinline }
        """
    quote
        Base.llvmcall(($ir, "entry"), Vec{$W,$T},
                      Tuple{Vec{$W,$T}, Vec{$W,$T}, Vec{$W,$T}}, a, b, c)
    end
end

# AVX-512 lane count for a supported type. Float64→8, Float32→16.
_avx512_W(::Type{Float64}) = 8
_avx512_W(::Type{Float32}) = 16

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
    AVX512Kernel{MR,NR,T}()

Explicit AVX-512 microkernel with NR vector accumulators per row group
(MR/W groups, where W is the lane count for `T`). Body emitted by
`@generated`. Requires `StridedMatrix{T}` for the SIMD path; falls back
to `ScalarKernel{MR,NR}` for other matrix types.

Examples:
    AVX512Kernel{8, 24, Float64}()   # MKL-style 8×24 (default)
    AVX512Kernel{16,14, Float64}()   # BLIS skx-style 16×14
    AVX512Kernel{8, 14, Float64}()
"""
struct AVX512Kernel{MR,NR,T} <: AbstractKernel end

# Backwards-compatible alias.
const AVX512F64Kernel = AVX512Kernel{8,24,Float64}

mr(::ScalarKernel{MR,NR}) where {MR,NR}        = MR
nr(::ScalarKernel{MR,NR}) where {MR,NR}        = NR
mr(::AVX512Kernel{MR,NR,T}) where {MR,NR,T}    = MR
nr(::AVX512Kernel{MR,NR,T}) where {MR,NR,T}    = NR

default_kernel(::Type{Float64}) = AVX512Kernel{8,24,Float64}()
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

# AVX-512 SIMD: cap KC so NR × KC × sizeof(T) ≤ ~28 KiB.
function kc_block(::AVX512Kernel{MR,NR,T}) where {MR,NR,T}
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

# ─── Microkernel: AVX512Kernel (parametric, JIT-unrolled SIMD) ───────────

# Non-strided fallback: route through the scalar kernel.
@inline function _kernel!(::AVX512Kernel{MR,NR,T}, C::AbstractMatrix{T},
                          Apack::Vector{T}, Bpack::Vector{T},
                          ao::Int, bo::Int, kc::Int, ci::Int, cj::Int,
                          mr_::Int, nr_::Int, α::T, v::Val) where {MR,NR,T}
    return _kernel!(ScalarKernel{MR,NR}(), C, Apack, Bpack, ao, bo, kc,
                    ci, cj, mr_, nr_, α, v)
end

@generated function _kernel!(::AVX512Kernel{MR,NR,T}, C::StridedMatrix{T},
                              Apack::Vector{T}, Bpack::Vector{T},
                              ao::Int, bo::Int, kc::Int, ci::Int, cj::Int,
                              mr_::Int, nr_::Int, α::T,
                              ::Val{full}) where {MR,NR,T,full}
    W = _avx512_W(T)
    MR % W == 0 ||
        throw(ArgumentError("AVX512Kernel: MR=$MR must be a multiple of W=$W for $T"))
    rows = MR ÷ W
    sz   = sizeof(T)
    Vty  = :(Vec{$W,$T})

    # Accumulator init
    init = Expr[]
    for r in 1:rows, j in 1:NR
        push!(init, :( $(Symbol("c_", r, "_", j)) = _vzero($Vty) ))
    end

    # Inner k-loop body
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

    # Vector writeback (full tile)
    write_full = Expr[]
    for j in 1:NR, r in 1:rows
        c   = Symbol("c_", r, "_", j)
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
