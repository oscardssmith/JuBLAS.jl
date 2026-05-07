# Shared infrastructure for JuBLAS:
#   - SIMD primitives not yet in SIMD.jl (`prefix_mask` and friends)
#   - Runtime CPU detection (cache sizes, SIMD width) via CpuId.jl
#   - Kernel type definitions (`AbstractKernel`, `ScalarKernel`, `SIMDKernel`)
#     + default kernel selection
#   - Cache-tuned block-size traits (`mc_block`, `kc_block`, `nc_block`)
#   - Misc helpers (`_scale!`)
#
# Real-specific gemm code lives in `gemm.jl`; complex-specific code in
# `gemm_complex.jl`. Both depend on the type and trait definitions here.

# ─── SIMD primitives (vendored from a pending SIMD.jl PR) ─────────────────
#
# Once the PR (https://github.com/eschnett/SIMD.jl/pull/XXX) lands and a
# release is tagged, the three definitions below can be deleted and `using
# SIMD` will pick up the upstream versions.

"""
    prefix_mask(::Val{N}, n::Integer) -> Vec{N,Bool}

Mask whose first `min(max(n, 0), N)` lanes are `true` and remaining lanes
are `false` — the canonical "vectorized-loop tail" mask. Lowers to a
small integer-ALU + `kmovd` sequence on AVX-512 (avoiding the heavy
vector compare LLVM emits for the lanewise `Vec{N,Bool}(ntuple(...))`
idiom).
"""
@generated function prefix_mask(::Val{N}, n::Int64) where {N}
    @assert N in (1, 2, 4, 8, 16, 32, 64) "prefix_mask: N must be a power of 2 ≤ 64"
    # Clamp `n` into [0, N] via smax then smin. Need both: `smin` alone
    # leaves negatives untouched (smin(-11, 16) = -11), and `umin` alone
    # treats negatives as huge unsigned (returns N → all lanes true) —
    # both wrong for callers that subtract a row offset and may go negative.
    #
    # `fshr.i64(0, -1, k)` interprets shift count `k` modulo 64, so a
    # `k == 64` (i.e., `clamped == 0`) collapses to `k == 0` and returns
    # `-1` instead of `0`. Guard with a `select` on the `clamped == 0` case.
    ir = """
        define <$N x i8> @entry(i64 %n) #0 {
        top:
            %nonneg = call i64 @llvm.smax.i64(i64 %n, i64 0)
            %clamped = call i64 @llvm.smin.i64(i64 %nonneg, i64 $N)
            %shift_amt = sub i64 64, %clamped
            %ones = sub i64 0, 1
            %shifted = call i64 @llvm.fshr.i64(i64 0, i64 %ones, i64 %shift_amt)
            %is_zero = icmp eq i64 %clamped, 0
            %bits64 = select i1 %is_zero, i64 0, i64 %shifted
            %bits = trunc i64 %bits64 to i$N
            %m1 = bitcast i$N %bits to <$N x i1>
            %m8 = zext <$N x i1> %m1 to <$N x i8>
            ret <$N x i8> %m8
        }
        declare i64 @llvm.smax.i64(i64, i64)
        declare i64 @llvm.smin.i64(i64, i64)
        declare i64 @llvm.fshr.i64(i64, i64, i64)
        attributes #0 = { alwaysinline }
        """
    quote
        $(Expr(:meta, :inline))
        nt = Base.llvmcall(($ir, "entry"), NTuple{$N, VecElement{Bool}},
                            Tuple{Int64}, n)
        Vec{$N,Bool}(nt)
    end
end

@inline prefix_mask(v::Val, n::Integer) = prefix_mask(v, Int64(n))

"""
    vload_prefix(::Type{Vec{N,T}}, ptr::Ptr{T}, n::Integer) -> Vec{N,T}

Load the first `min(max(n, 0), N)` consecutive `T`s from `ptr` into the
low lanes of a `Vec{N,T}`; remaining lanes are zero. Equivalent to
`vload(Vec{N,T}, ptr, prefix_mask(Val(N), n))`.
"""
@inline vload_prefix(::Type{Vec{N,T}}, ptr::Ptr{T}, n::Integer) where {N,T} =
    vload(Vec{N,T}, ptr, prefix_mask(Val(N), Int64(n)))

"""
    vstore_prefix!(v::Vec{N,T}, ptr::Ptr{T}, n::Integer) -> Nothing

Store the first `min(max(n, 0), N)` lanes of `v` to `ptr`; remaining
lanes are not written. Equivalent to
`vstore(v, ptr, prefix_mask(Val(N), n))`.
"""
@inline vstore_prefix!(v::Vec{N,T}, ptr::Ptr{T}, n::Integer) where {N,T} =
    vstore(v, ptr, prefix_mask(Val(N), Int64(n)))

# ─── Runtime CPU detection ────────────────────────────────────────────────
#
# CPUID parses are cheap individually but called several times per `gemm!`
# (3 cache levels + simd width), and a try/catch around each one is not
# free. At N=50 the matmul itself is ~5 µs, so per-call CPUID cost shows
# up in the profile — measured ~3 µs of overhead in `_gemm!` outside the
# packing/kernel phases, much of it from these calls.
#
# Memoize via a sentinel `Ref`, populated on first call. Avoids `__init__`
# (which fights precompile) and keeps the hot path to a Ref load + branch.
# CpuId is x86-only and may throw on hypervisors that hide cache info;
# fall back to Skylake-class defaults.

const _CACHE_SIZES_CACHE = Ref{NTuple{3,Int}}((0, 0, 0))
const _SIMD_BYTES_CACHE  = Ref{Int}(-1)

@inline function _cache_sizes()
    cs = _CACHE_SIZES_CACHE[]
    cs[1] == 0 ? _detect_cache_sizes!() : cs
end

@noinline function _detect_cache_sizes!()
    cs = try
        raw = CpuId.cachesize()
        l1 = length(raw) >= 1 && raw[1] > 0 ? Int(raw[1]) : 32 * 1024
        l2 = length(raw) >= 2 && raw[2] > 0 ? Int(raw[2]) : 1024 * 1024
        l3 = length(raw) >= 3 && raw[3] > 0 ? Int(raw[3]) : 16 * 1024 * 1024
        (l1, l2, l3)
    catch
        (32 * 1024, 1024 * 1024, 16 * 1024 * 1024)
    end
    _CACHE_SIZES_CACHE[] = cs
    return cs
end

@inline function _simd_bytes()
    sb = _SIMD_BYTES_CACHE[]
    sb >= 0 ? sb : _detect_simd_bytes!()
end

@noinline function _detect_simd_bytes!()
    # Env-var override for benchmarking AVX2/SSE2 paths on AVX-512 hardware:
    # `JULIA_JUBLAS_SIMD_BYTES=32 julia --cpu-target=haswell ...` tunes the
    # AVX2 branch without needing different silicon.
    sb = let env = get(ENV, "JULIA_JUBLAS_SIMD_BYTES", "")
        isempty(env) ? (try; Int(CpuId.simdbytes()); catch; 0; end) :
                       parse(Int, env)
    end
    _SIMD_BYTES_CACHE[] = sb
    return sb
end

# ─── Kernel types ─────────────────────────────────────────────────────────

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

For `T <: Real`, kernel methods are in `gemm.jl`; for `T <: Complex`,
in `gemm_complex.jl`.

Examples:
    SIMDKernel{8,  8, 24, Float64}()       # AVX-512 real
    SIMDKernel{8, 16, 14, Float64}()       # AVX-512 real, BLIS skx-style
    SIMDKernel{4,  8,  6, Float64}()       # AVX2 real
    SIMDKernel{8,  8, 12, ComplexF64}()    # AVX-512 complex
    SIMDKernel{16,16, 12, ComplexF32}()    # AVX-512 complex
"""
struct SIMDKernel{W,MR,NR,T} <: AbstractKernel end

mr(::ScalarKernel{MR,NR})   where {MR,NR}     = MR
nr(::ScalarKernel{MR,NR})   where {MR,NR}     = NR
mr(::SIMDKernel{W,MR,NR,T}) where {W,MR,NR,T} = MR
nr(::SIMDKernel{W,MR,NR,T}) where {W,MR,NR,T} = NR

# Shape-aware default. Per-eltype methods can specialize on (M, N, K) —
# e.g. Float64 AVX-512 switches to a smaller-NR kernel when N is too
# small for the default NR to tile cleanly. The generic fallback ignores
# shape and returns the scalar kernel.
default_kernel(::Type{T}, M::Int, N::Int, K::Int) where {T} = ScalarKernel{8, 6}()
# 1-arg convenience: no shape hint → call the 4-arg form with `typemax`,
# which lands in the wide-N branch for any specialized eltype.
default_kernel(::Type{T}) where {T} =
    default_kernel(T, typemax(Int), typemax(Int), typemax(Int))

# ─── Block sizes ──────────────────────────────────────────────────────────
#
# `mc_block`: rows of A panel held in L2. Must be a multiple of MR so the
#             macrokernel's ir-loop has no edge panels in the common case.
# `kc_block`: depth of A/B panels. For SIMD kernels we size KC so the B
#             micropanel (NR × KC × T) lives in L1d (A is streamed).
# `nc_block`: width of B slab held in L3.
#
# Each `*_block` method takes a `(L1d, L2, L3)` cache-size tuple. Callers
# (`gemm!`, `gemm_workspace`) query `_cache_sizes()` once and thread the
# tuple through, so a single `gemm!` call costs one CpuId trip.

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
# Each block is rounded to a clean multiple of the relevant register tile
# dimension so the macrokernel's loops have no edge in the bulk case. The
# byte arithmetic uses `sizeof(T)`, which gives `2 * sizeof(TR)` for
# `T = Complex{TR}` automatically — no separate complex methods needed.

function kc_block(::SIMDKernel{W,MR,NR,T}, caches=_cache_sizes()) where {W,MR,NR,T}
    l1     = caches[1]
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

# ─── Misc helpers ─────────────────────────────────────────────────────────

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
