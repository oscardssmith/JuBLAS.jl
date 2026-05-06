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

# `prefix_mask`, `vload_prefix`, `vstore_prefix!` come from `utils.jl`
# (vendored from a pending SIMD.jl PR). Used to mask the row-direction edge
# of the MR×NR writeback tile.

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

# Runtime CPU detection. Each call issues CPUID and parses the result —
# ~hundreds of cycles, run a handful of times per `gemm!` call, lost in
# the noise of a millisecond-scale matmul. CpuId is x86-only and may throw
# on hypervisors that hide cache info; fall back to Skylake-class defaults.
function _cache_sizes()
    try
        cs = CpuId.cachesize()
        l1 = length(cs) >= 1 && cs[1] > 0 ? Int(cs[1]) : 32 * 1024
        l2 = length(cs) >= 2 && cs[2] > 0 ? Int(cs[2]) : 1024 * 1024
        l3 = length(cs) >= 3 && cs[3] > 0 ? Int(cs[3]) : 16 * 1024 * 1024
        return (l1, l2, l3)
    catch
        return (32 * 1024, 1024 * 1024, 16 * 1024 * 1024)
    end
end
_simd_bytes() = try; Int(CpuId.simdbytes()); catch; 0; end

# Pick the strongest microkernel the host CPU can run. AVX-512 (zmm, 64 B),
# then AVX2 (ymm, 32 B), then SSE2 (xmm, 16 B), else scalar. The MR×NR shapes
# come from the bench sweep — these are the configs that consistently won on
# Skylake-class hardware.
function default_kernel(::Type{Float64})
    sb = _simd_bytes()
    sb >= 64 ? SIMDKernel{8, 16, 14, Float64}() :
    sb >= 32 ? SIMDKernel{4,  8,  6, Float64}() :
    sb >= 16 ? SIMDKernel{2,  2,  4, Float64}() :
               ScalarKernel{8, 6}()
end
function default_kernel(::Type{Float32})
    sb = _simd_bytes()
    sb >= 64 ? SIMDKernel{16, 32, 14, Float32}() :
    sb >= 32 ? SIMDKernel{8,  16,  6, Float32}() :
    sb >= 16 ? SIMDKernel{4,   4,  6, Float32}() :
               ScalarKernel{16, 6}()
end
default_kernel(::Type{T}) where {T} = ScalarKernel{8, 6}()

# ─── Block sizes (per-kernel traits) ──────────────────────────────────────
#
# `mc_block`: rows of A panel held in L2. Must be a multiple of MR so the
#             macrokernel's ir-loop has no edge panels in the common case.
# `kc_block`: depth of A/B panels. For SIMD kernels we size KC so the B
#             micropanel (NR × KC × T) lives in L1d (A is streamed).
# `nc_block`: width of B slab held in L3.

# Each `*_block` method takes a `(L1d, L2, L3)` cache-size tuple. Callers
# (`gemm!`, `gemm_workspace`) query `_cache_sizes()` once and thread the
# tuple through, so a single `gemm!` call costs one CpuId trip.
#
# ScalarKernel / AbstractKernel fallback: no cache-aware tuning — used only
# for the non-strided-C path, which isn't perf-critical.
mc_block(k::AbstractKernel, _caches=_cache_sizes()) = cld(72, mr(k)) * mr(k)
kc_block(::AbstractKernel,  _caches=_cache_sizes()) = 256
nc_block(::AbstractKernel,  _caches=_cache_sizes()) = 4080

# SIMDKernel block sizes follow the Goto/BLIS cache hierarchy:
#
#   KC: B micropanel (NR × KC × T) is hot in L1d throughout one macrokernel
#       call (reused across MC/MR ir-iterations). A is *streamed* one
#       column-of-MR per k-iter, so only a few A cache lines are hot at any
#       moment — we don't budget the full A microtile against L1d. Take 3/4
#       of L1d to leave room for the streamed A lines, stack, and writeback.
#
#   MC: A panel (MC × KC × T) is hot in L2 across all NR-stride steps of the
#       jr loop. Budget half of L2, leaving room for the B slab to coexist
#       (it also funnels through L2 from L3) and for system noise.
#
#   NC: B slab (KC × NC × T) is hot in L3 across all MC-stride steps of the
#       ic loop. Budget half of L3 to share with A and any co-runners.
#
# Each block is rounded to be a clean multiple of the relevant register
# tile dimension so the macrokernel's loops have no edge in the bulk case.

function kc_block(::SIMDKernel{W,MR,NR,T}, caches=_cache_sizes()) where {W,MR,NR,T}
    l1 = caches[1]
    budget = (l1 * 3) ÷ 4
    raw    = budget ÷ (NR * sizeof(T))
    return clamp(raw - raw % 8, 64, 512)
end

function mc_block(k::SIMDKernel{W,MR,NR,T}, caches=_cache_sizes()) where {W,MR,NR,T}
    kc        = kc_block(k, caches)
    l2        = caches[2]
    target_mc = max((l2 ÷ 2) ÷ (kc * sizeof(T)), MR)
    return cld(target_mc, MR) * MR
end

function nc_block(k::SIMDKernel{W,MR,NR,T}, caches=_cache_sizes()) where {W,MR,NR,T}
    kc        = kc_block(k, caches)
    l3        = caches[3]
    target_nc = max((l3 ÷ 2) ÷ (kc * sizeof(T)), NR)
    return (target_nc ÷ NR) * NR
end

# ─── Main entry point ─────────────────────────────────────────────────────

"""
    gemm_workspace(::Type{T}, kernel) -> (Apack, Bpack)

Allocate the pack buffers a `kernel`-shaped `gemm!` call needs. Pass them
back via the `Apack`/`Bpack` keywords to amortize allocation across many
calls and keep the buffers warm in cache. Sized for the kernel's full
block dimensions (`MC×KC` for A, `NC×KC` for B); the same buffers are valid
for any problem size that fits.
"""
function gemm_workspace(::Type{T}, kernel::AbstractKernel) where {T}
    MR_ = mr(kernel); NR_ = nr(kernel)
    caches = _cache_sizes()
    MC_ = mc_block(kernel, caches)
    KC_ = kc_block(kernel, caches)
    NC_ = nc_block(kernel, caches)
    Apack = Vector{T}(undef, cld(MC_, MR_) * MR_ * KC_)
    Bpack = Vector{T}(undef, cld(NC_, NR_) * NR_ * KC_)
    return (Apack, Bpack)
end

function gemm!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T},
               α = true, β = false;
               kernel::AbstractKernel = default_kernel(T),
               Apack::Union{Vector{T},Nothing} = nothing,
               Bpack::Union{Vector{T},Nothing} = nothing) where {T}
    M, N = size(C)
    size(A, 1) == M           || throw(DimensionMismatch("size(A,1) ≠ size(C,1)"))
    size(B, 2) == N           || throw(DimensionMismatch("size(B,2) ≠ size(C,2)"))
    size(A, 2) == size(B, 1)  || throw(DimensionMismatch("size(A,2) ≠ size(B,1)"))
    K = size(A, 2)

    _scale!(C, β)
    (M == 0 || N == 0 || K == 0) && return C

    MR_ = mr(kernel)
    NR_ = nr(kernel)
    caches = _cache_sizes()
    MC_ = mc_block(kernel, caches)
    KC_ = kc_block(kernel, caches)
    NC_ = nc_block(kernel, caches)

    mc_max = min(MC_, M)
    kc_max = min(KC_, K)
    nc_max = min(NC_, N)
    apack_sz = cld(mc_max, MR_) * MR_ * kc_max
    bpack_sz = cld(nc_max, NR_) * NR_ * kc_max
    Apack === nothing && (Apack = Vector{T}(undef, apack_sz))
    Bpack === nothing && (Bpack = Vector{T}(undef, bpack_sz))
    length(Apack) >= apack_sz ||
        throw(ArgumentError("Apack too small: have $(length(Apack)), need $apack_sz"))
    length(Bpack) >= bpack_sz ||
        throw(ArgumentError("Bpack too small: have $(length(Bpack)), need $bpack_sz"))
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

# `@generated` with explicit `Vec{MR,T}` load/store on the fast path. This
# bypasses LLVM's cost-model preference for ymm-versioning + scalar tails:
# we hand it a single `<MR × T>` value, which lowers to one (or `MR/W`)
# zmm-width move on AVX-512 targets, no memcheck, no tail.
# Slow path: edge panel with partial copy + zero fill.
@generated function _pack_A!(Apack::Vector{T}, A::AbstractMatrix{T},
                              ic::Int, pc::Int, mc::Int, kc::Int,
                              ::Val{MR}) where {T,MR}
    sz = sizeof(T)
    quote
        $(Expr(:meta, :inline))
        z = zero($T)
        npanels = cld(mc, $MR)
        GC.@preserve Apack A begin
            pAp = pointer(Apack)
            pA  = pointer(A)
            ldA = stride(A, 2)
            @inbounds for p in 0:npanels-1
                rs   = p * $MR
                rmax = min($MR, mc - rs)
                base_panel = p * $MR * kc
                if rmax == $MR
                    # Bulk MR-element copy via raw `NTuple{MR,VecElement{T}}`
                    # cast. Works for any `isbitstype(T)` (including
                    # `ComplexF64`), unlike `SIMD.Vec` which is restricted to
                    # the scalar number types.
                    for k in 0:kc-1
                        src_off = ((ic + rs - 1) + (pc + k - 1) * ldA) * $sz
                        dst_off = (base_panel + k * $MR) * $sz
                        nt = unsafe_load(Ptr{NTuple{$MR, VecElement{$T}}}(pA + src_off))
                        unsafe_store!(Ptr{NTuple{$MR, VecElement{$T}}}(pAp + dst_off), nt)
                    end
                else
                    for k in 0:kc-1
                        base = base_panel + k * $MR
                        for i in 1:rmax
                            Apack[base + i] = A[ic + rs + i - 1, pc + k]
                        end
                        for i in rmax+1:$MR
                            Apack[base + i] = z
                        end
                    end
                end
            end
        end
        return Apack
    end
end

# Pointer-based packing for B with the LLVM loop vectorizer **disabled** on
# the inner k-loops. Without that hint, LLVM vectorizes across k iterations:
# loads end up contiguous (good) but stores get a stride of `NR*sizeof(T)`
# (= 112 B for Float64, NR=14), forcing `vscatterqpd` that loses badly. The
# vectorizer also versions the loop with O(NR²) pairwise alias checks,
# blowing the GPR budget and causing ~50 spills/reloads in the prologue.
# With vectorization off, the fast path becomes a clean walking-pointer
# loop: 14× `vmovsd` load + 14× `vmovsd` store per k, no scatter, no
# memcheck, no scratch-stack churn.
@generated function _pack_B!(Bpack::Vector{T}, B::AbstractMatrix{T},
                              pc::Int, jc::Int, kc::Int, nc::Int,
                              ::Val{NR}) where {T,NR}
    sz = sizeof(T)
    fast_copy = [:( unsafe_store!(pBp, unsafe_load(pB + ($(j-1)) * ldB_b), $j) ) for j in 1:NR]
    novec = Expr(:loopinfo, (Symbol("llvm.loop.vectorize.enable"), false))
    quote
        $(Expr(:meta, :inline))
        z = zero($T)
        npanels = cld(nc, $NR)
        GC.@preserve Bpack B begin
            pBp0 = pointer(Bpack)
            pB0  = pointer(B)
            ldB  = stride(B, 2)
            ldB_b = ldB * $sz
            @inbounds for p in 0:npanels-1
                cs   = p * $NR
                cmax = min($NR, nc - cs)
                base_panel = p * $NR * kc
                col_off_b = ((pc - 1) + (jc + cs - 1) * ldB) * $sz
                if cmax == $NR
                    for k in 0:kc-1
                        pB  = pB0  + col_off_b + k * $sz
                        pBp = pBp0 + (base_panel + k * $NR) * $sz
                        $(fast_copy...)
                        $novec
                    end
                else
                    for k in 0:kc-1
                        base = base_panel + k * $NR
                        for j in 1:cmax
                            Bpack[base + j] = B[pc + k, jc + cs + j - 1]
                            $novec
                        end
                        for j in cmax+1:$NR
                            Bpack[base + j] = z
                            $novec
                        end
                        $novec
                    end
                end
            end
        end
        return Bpack
    end
end

# ─── Macrokernel ──────────────────────────────────────────────────────────

function _macrokernel!(kernel::AbstractKernel, C::AbstractMatrix{T},
                       Apack::Vector{T}, Bpack::Vector{T},
                       ic::Int, jc::Int, mc::Int, nc::Int, kc::Int, α::T) where {T}
    MR_ = mr(kernel)
    NR_ = nr(kernel)
    @inbounds for jr in 0:NR_:nc-1
        nr_ = min(NR_, nc - jr)
        bo = jr * kc
        for ir in 0:MR_:mc-1
            mr_ = min(MR_, mc - ir)
            ao = ir * kc
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
        push!(init, :( $(Symbol("c_", r, "_", j)) = zero($Vty) ))
    end

    # Inner k-loop body. `pA`/`pB` are extracted once at the top of the kernel
    # so the loop sees byte-offset arithmetic only.
    inner = Expr[]
    for r in 1:rows
        a   = Symbol("a_", r)
        off = (r - 1) * W
        push!(inner, :( $a = vload($Vty, pA + (ao + k * $MR + $off) * $sz) ))
    end
    for j in 1:NR
        bv = Symbol("bv_", j)
        push!(inner, :( $bv = $Vty(unsafe_load(pB, bo + k * $NR + $j)) ))
        for r in 1:rows
            c = Symbol("c_", r, "_", j); a = Symbol("a_", r)
            push!(inner, :( $c = muladd($a, $bv, $c) ))
        end
    end

    # Vector writeback (full tile). `pC`/`ldc` extracted once outside; the
    # column address is walked with `pCj += ldc*sz` instead of recomputing
    # `(cj+j-1)*ldc*sz` per column. Otherwise the macrokernel hoists NR
    # column indices out of the ir-loop and spills them all to the stack
    # (only 16 GPRs), then reloads each one for an `imul` per column.
    write_full = Expr[]
    push!(write_full, :( col_stride_b = ldc * $sz ))
    push!(write_full, :( pCj = pC + ((cj - 1) * ldc + (ci - 1)) * $sz ))
    for j in 1:NR
        for r in 1:rows
            c    = Symbol("c_", r, "_", j)
            roff = (r - 1) * W
            poff = roff * sz
            p    = poff == 0 ? :pCj : :( pCj + $poff )
            push!(write_full, quote
                let p = $p
                    vstore(muladd(αv, $c, vload($Vty, p)), p)
                end
            end)
        end
        if j < NR
            push!(write_full, :( pCj = pCj + col_stride_b ))
        end
    end

    # Masked vector edge writeback. Same pCj walk as the full-tile path, plus
    # the row-mask is hoisted (depends only on `mr_` and row group `r`, not
    # on `j`). Columns past `nr_` are skipped at runtime via the
    # compile-time-literal `j <= nr_` check; pCj is still advanced through
    # the skipped columns (cheap and lets LLVM keep one rolling register).
    write_edge = Expr[]
    for r in 1:rows
        roff   = (r - 1) * W
        mask_r = Symbol("mask_", r)
        push!(write_edge, :( $mask_r = prefix_mask(Val($W), mr_ - $roff) ))
    end
    push!(write_edge, :( col_stride_b = ldc * $sz ))
    push!(write_edge, :( pCj = pC + ((cj - 1) * ldc + (ci - 1)) * $sz ))
    for j in 1:NR
        body = Expr[]
        for r in 1:rows
            c      = Symbol("c_", r, "_", j)
            mask_r = Symbol("mask_", r)
            roff   = (r - 1) * W
            poff   = roff * sz
            p      = poff == 0 ? :pCj : :( pCj + $poff )
            push!(body, quote
                if any($mask_r)
                    let p = $p
                        cur = vload($Vty, p, $mask_r)
                        vstore(muladd(αv, $c, cur), p, $mask_r)
                    end
                end
            end)
        end
        push!(write_edge, quote
            if $j <= nr_
                $(body...)
            end
        end)
        if j < NR
            push!(write_edge, :( pCj = pCj + col_stride_b ))
        end
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
            αv  = $Vty(α)
            if $full
                $(write_full...)
            else
                @inbounds begin $(write_edge...) end
            end
        end
        return nothing
    end
end
