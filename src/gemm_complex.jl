# Complex GEMM via `SIMDKernel{W,MR,NR,Complex{TR}}`. The Re and Im
# components of each panel are packed into separate sub-panels (struct-of-
# arrays); the inner k-loop synthesizes one complex FMA from four real FMAs.
#
# Memory layout per packed panel (for one MR-row block, kc deep):
#
#     [ Re of k=0, MR floats ][ Im of k=0, MR floats ]
#     [ Re of k=1, MR floats ][ Im of k=1, MR floats ]
#     ...
#
# Total panel size in real elements: 2 × MR × kc. Same shape for Bpack with
# NR replacing MR. This layout puts Re/Im for the same k adjacent, so the
# inner loop's two A-loads (Are, Aim) hit consecutive cache lines.

# ─── Default kernel selection ─────────────────────────────────────────────

# AVX-512 defaults are bench-picked from `bench/complex_shape_sweep.jl`:
# rows=2 tiles win because the larger MR (= 2W) gets more A reuse per k-iter,
# and the smaller NR (`*sizeof(Complex{TR})` budget for L1d) lets KC grow to
# its 512 clamp, halving pack overhead. AVX2/SSE2 fallbacks stay rows=1
# because their 16-register file can't host the rows=2 accumulator tile.
function default_kernel(::Type{ComplexF64})
    sb = _simd_bytes()
    sb >= 64 ? SIMDKernel{8, 16,  6, ComplexF64}() :   # AVX-512 rows=2
    sb >= 32 ? SIMDKernel{4,  4,  6, ComplexF64}() :   # AVX2 ymm
    SIMDKernel{2,  2,  4, ComplexF64}()                # SSE2 xmm
end
function default_kernel(::Type{ComplexF32})
    sb = _simd_bytes()
    sb >= 64 ? SIMDKernel{16, 32, 4, ComplexF32}() :   # AVX-512 rows=2
    sb >= 32 ? SIMDKernel{8,   8, 6, ComplexF32}() :   # AVX2 ymm
    SIMDKernel{4,   4, 6, ComplexF32}()                # SSE2 xmm
end

# Block sizes (`mc_block` / `kc_block` / `nc_block`) for `SIMDKernel{...,
# Complex{TR}}` are inherited from the generic `SIMDKernel{W,MR,NR,T}`
# methods in `utils.jl`. The byte arithmetic via `sizeof(T)` already gives
# `2 × sizeof(TR)` for `T = Complex{TR}`, so the formulas collapse cleanly.
#
# We tried complex-specific overrides (smaller KC, smaller MC) based on
# `bench/complex_block_sweep.jl` outputs, but those numbers turned out to
# track the sweep's per-call buffer allocation pattern rather than the
# production scenario where `Apack`/`Bpack` are persistent across many
# `gemm!` calls. The generic real-gemm formulas match the persistent-buffer
# `perf_bench_complex.jl` peak, so we keep them.

# ─── Packing ──────────────────────────────────────────────────────────────

# `_pack_A!` for `A::Matrix{Complex{TR}}` into `Apack::Vector{TR}` with
# Re/Im sub-panel layout described at the top of the file.
@generated function _pack_A!(Apack::Vector{TR}, A::AbstractMatrix{Complex{TR}},
                              ic::Int, pc::Int, mc::Int, kc::Int,
                              ::Val{MR}) where {TR<:Real, MR}
    sz = sizeof(TR)
    quote
        $(Expr(:meta, :inline))
        z = zero($TR)
        npanels = cld(mc, $MR)
        GC.@preserve Apack A begin
            pAp = pointer(Apack)
            pA  = Ptr{$TR}(pointer(A))
            ldA = stride(A, 2)              # in complex elements
            @inbounds for p in 0:npanels-1
                rs    = p * $MR
                rmax  = min($MR, mc - rs)
                base  = p * 2 * $MR * kc
                if rmax == $MR
                    # Fast path: full MR-row tile. Each k iter touches 2*MR
                    # consecutive `TR` values in column-major A and writes them
                    # de-interleaved into Apack.
                    for k in 0:kc-1
                        src_off_b = 2 * $sz * ((ic + rs - 1) + (pc + k - 1) * ldA)
                        dst_re_b  = $sz * (base + k * 2 * $MR)
                        dst_im_b  = dst_re_b + $sz * $MR
                        for i in 0:$MR-1
                            re = unsafe_load(pA + src_off_b + 2*$sz*i)
                            im = unsafe_load(pA + src_off_b + 2*$sz*i + $sz)
                            unsafe_store!(pAp + dst_re_b + $sz*i, re)
                            unsafe_store!(pAp + dst_im_b + $sz*i, im)
                        end
                    end
                else
                    for k in 0:kc-1
                        base_re = base + k * 2 * $MR
                        base_im = base_re + $MR
                        for i in 1:rmax
                            c = A[ic + rs + i - 1, pc + k]
                            Apack[base_re + i] = real(c)
                            Apack[base_im + i] = imag(c)
                        end
                        for i in rmax+1:$MR
                            Apack[base_re + i] = z
                            Apack[base_im + i] = z
                        end
                    end
                end
            end
        end
        return Apack
    end
end

@generated function _pack_B!(Bpack::Vector{TR}, B::AbstractMatrix{Complex{TR}},
                              pc::Int, jc::Int, kc::Int, nc::Int,
                              ::Val{NR}) where {TR<:Real, NR}
    sz = sizeof(TR)
    quote
        $(Expr(:meta, :inline))
        z = zero($TR)
        npanels = cld(nc, $NR)
        GC.@preserve Bpack B begin
            pBp = pointer(Bpack)
            pB  = Ptr{$TR}(pointer(B))
            ldB = stride(B, 2)              # in complex elements
            @inbounds for p in 0:npanels-1
                cs   = p * $NR
                cmax = min($NR, nc - cs)
                base = p * 2 * $NR * kc
                if cmax == $NR
                    for k in 0:kc-1
                        base_re = base + k * 2 * $NR
                        base_im = base_re + $NR
                        for j in 1:$NR
                            src_off_b = 2 * $sz * ((pc + k - 1) + (jc + cs + j - 2) * ldB)
                            re = unsafe_load(pB + src_off_b)
                            im = unsafe_load(pB + src_off_b + $sz)
                            unsafe_store!(pBp + $sz * (base_re + j - 1), re)
                            unsafe_store!(pBp + $sz * (base_im + j - 1), im)
                        end
                    end
                else
                    for k in 0:kc-1
                        base_re = base + k * 2 * $NR
                        base_im = base_re + $NR
                        for j in 1:cmax
                            c = B[pc + k, jc + cs + j - 1]
                            Bpack[base_re + j] = real(c)
                            Bpack[base_im + j] = imag(c)
                        end
                        for j in cmax+1:$NR
                            Bpack[base_re + j] = z
                            Bpack[base_im + j] = z
                        end
                    end
                end
            end
        end
        return Bpack
    end
end

# ─── Packing for transposed/adjoint complex inputs ────────────────────────
#
# `Transpose` packs are role-swap forwards (same identity used in
# `gemm.jl` for reals): packing `transpose(P)` as A is the same as
# packing `P` as B with the indices swapped, and vice versa.
#
# `Adjoint` would do the same forward but additionally needs to conjugate
# the imaginary sub-panels written to the pack buffer. We get that by
# (1) packing through the Transpose path and (2) sign-flipping every Im
# half of the resulting buffer. The pack buffer is roughly 2*MR*KC reals
# (or 2*NR*KC) per panel and is touched once per macrokernel call, so a
# second linear pass over it is dwarfed by the kc² FMA work in the
# kernel. (Inlining the conjugation into a custom @generated method
# would save this pass, but the saving is in the noise; reuse wins.)

@inline _pack_A!(Apack::Vector{TR},
                 A::Transpose{Complex{TR}, <:StridedMatrix{Complex{TR}}},
                 ic::Int, pc::Int, mc::Int, kc::Int,
                 v::Val) where {TR<:Real} =
    _pack_B!(Apack, parent(A), pc, ic, kc, mc, v)

@inline _pack_B!(Bpack::Vector{TR},
                 B::Transpose{Complex{TR}, <:StridedMatrix{Complex{TR}}},
                 pc::Int, jc::Int, kc::Int, nc::Int,
                 v::Val) where {TR<:Real} =
    _pack_A!(Bpack, parent(B), jc, pc, nc, kc, v)

# Negate every Im half of a packed buffer with `npanels` panels, each panel
# laid out as (Re k=0)(Im k=0)(Re k=1)(Im k=1)... with `lane` reals per half.
@inline function _conj_im_panels!(buf::Vector{TR}, npanels::Int, lane::Int, kc::Int) where {TR<:Real}
    @inbounds for p in 0:npanels-1
        base = p * 2 * lane * kc
        for k in 0:kc-1
            base_im = base + k * 2 * lane + lane
            for i in 1:lane
                buf[base_im + i] = -buf[base_im + i]
            end
        end
    end
    return buf
end

@inline function _pack_A!(Apack::Vector{TR},
                          A::Adjoint{Complex{TR}, <:StridedMatrix{Complex{TR}}},
                          ic::Int, pc::Int, mc::Int, kc::Int,
                          v::Val{MR}) where {TR<:Real, MR}
    _pack_B!(Apack, parent(A), pc, ic, kc, mc, v)
    _conj_im_panels!(Apack, cld(mc, MR), MR, kc)
    return Apack
end

@inline function _pack_B!(Bpack::Vector{TR},
                          B::Adjoint{Complex{TR}, <:StridedMatrix{Complex{TR}}},
                          pc::Int, jc::Int, kc::Int, nc::Int,
                          v::Val{NR}) where {TR<:Real, NR}
    _pack_A!(Bpack, parent(B), jc, pc, nc, kc, v)
    _conj_im_panels!(Bpack, cld(nc, NR), NR, kc)
    return Bpack
end

# ─── Microkernel: complex SIMD ───────────────────────────────────────────
# Inner loop per k:
#   load   Are[r], Aim[r]                    (2 vec loads per row group)
#   broadcast Bre[j], Bim[j]                 (folded into FMA on AVX-512)
#   cre[r,j] += Are[r] * Bre[j] - Aim[r] * Bim[j]
#   cim[r,j] += Are[r] * Bim[j] + Aim[r] * Bre[j]
# = 4 real FMAs per (r, j) per k. Lowers to `vfmadd231` / `vfnmadd231`
# pairs on AVX-512.
@generated function _kernel!(::SIMDKernel{W,MR,NR,Complex{TR}}, C::StridedMatrix{Complex{TR}},
                              Apack::Vector{TR}, Bpack::Vector{TR},
                              ao::Int, bo::Int, kc::Int, ci::Int, cj::Int,
                              mr_::Int, nr_::Int, α::Complex{TR},
                              ::Val{full}) where {W,MR,NR,TR<:Union{Float32,Float64},full}
    MR % W == 0 || throw(ArgumentError("SIMDKernel: MR=$MR must be a multiple of W=$W"))
    rows = MR ÷ W
    sz   = sizeof(TR)
    Vty  = :(Vec{$W,$TR})

    # Accumulator init: two vectors per cell (real, imaginary).
    init = Expr[]
    for r in 1:rows, j in 1:NR
        push!(init, :( $(Symbol("cre_", r, "_", j)) = zero($Vty) ))
        push!(init, :( $(Symbol("cim_", r, "_", j)) = zero($Vty) ))
    end

    inner = Expr[]
    for r in 1:rows
        roff = (r - 1) * W
        push!(inner, :( $(Symbol("are_", r)) = vload($Vty, pA + (ao + k * 2 * $MR + $roff) * $sz) ))
        push!(inner, :( $(Symbol("aim_", r)) = vload($Vty, pA + (ao + k * 2 * $MR + $MR + $roff) * $sz) ))
    end
    for j in 1:NR
        bre = Symbol("bre_", j); bim = Symbol("bim_", j)
        push!(inner, :( $bre = $Vty(unsafe_load(pB, bo + k * 2 * $NR + $j)) ))
        push!(inner, :( $bim = $Vty(unsafe_load(pB, bo + k * 2 * $NR + $NR + $j)) ))
        for r in 1:rows
            cre = Symbol("cre_", r, "_", j); cim = Symbol("cim_", r, "_", j)
            are = Symbol("are_", r);          aim = Symbol("aim_", r)
            # cre += are*bre - aim*bim
            push!(inner, :( $cre = muladd( $are, $bre, $cre) ))
            push!(inner, :( $cre = muladd(-$aim, $bim, $cre) ))
            # cim += are*bim + aim*bre
            push!(inner, :( $cim = muladd( $are, $bim, $cim) ))
            push!(inner, :( $cim = muladd( $aim, $bre, $cim) ))
        end
    end

    # Writeback: scalar element-wise. C is interleaved Complex; SIMD
    # interleaved-store would need a shuffle pair to merge cre/cim back into
    # the layout of C. The MR×NR×W stores per kernel call are tiny relative
    # to the FMA loop (~kc× more work), so a scalar loop is fine for now.
    write_full = Expr[]
    for j in 1:NR
        for r in 1:rows
            roff = (r - 1) * W
            cre  = Symbol("cre_", r, "_", j); cim = Symbol("cim_", r, "_", j)
            push!(write_full, quote
                let cre_v = $cre, cim_v = $cim
                    @inbounds for i in 0:$W-1
                        c_off_b = 2 * $sz * ((ci + $roff + i - 1) + (cj + $(j-1) - 1) * ldc)
                        oldre = unsafe_load(pC + c_off_b)
                        oldim = unsafe_load(pC + c_off_b + $sz)
                        re    = cre_v[i+1]
                        im    = cim_v[i+1]
                        unsafe_store!(pC + c_off_b,        oldre + αre*re - αim*im)
                        unsafe_store!(pC + c_off_b + $sz,  oldim + αre*im + αim*re)
                    end
                end
            end)
        end
    end

    write_edge = Expr[]
    for j in 1:NR
        for r in 1:rows
            roff = (r - 1) * W
            cre  = Symbol("cre_", r, "_", j); cim = Symbol("cim_", r, "_", j)
            push!(write_edge, quote
                if $j <= nr_
                    let cre_v = $cre, cim_v = $cim
                        @inbounds for i in 0:$W-1
                            if $roff + i < mr_
                                c_off_b = 2 * $sz * ((ci + $roff + i - 1) + (cj + $(j-1) - 1) * ldc)
                                oldre = unsafe_load(pC + c_off_b)
                                oldim = unsafe_load(pC + c_off_b + $sz)
                                re    = cre_v[i+1]
                                im    = cim_v[i+1]
                                unsafe_store!(pC + c_off_b,        oldre + αre*re - αim*im)
                                unsafe_store!(pC + c_off_b + $sz,  oldim + αre*im + αim*re)
                            end
                        end
                    end
                end
            end)
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

            ldc = stride(C, 2)
            pC  = Ptr{$TR}(pointer(C))
            αre = real(α); αim = imag(α)
            if $full
                $(write_full...)
            else
                $(write_edge...)
            end
        end
        return nothing
    end
end

# AbstractMatrix fallback: pack/macrokernel still needs strided pointer math,
# but if `C` isn't strided we have nowhere to point. Defer to the scalar
# kernel.
@inline function _kernel!(k::SIMDKernel{W,MR,NR,Complex{TR}}, C::AbstractMatrix{Complex{TR}},
                          Apack::Vector{TR}, Bpack::Vector{TR},
                          ao::Int, bo::Int, kc::Int, ci::Int, cj::Int,
                          mr_::Int, nr_::Int, α::Complex{TR},
                          v::Val) where {W,MR,NR,TR<:Real}
    error("SIMDKernel{...,Complex} requires StridedMatrix C; got $(typeof(C))")
end

# ─── Macrokernel ──────────────────────────────────────────────────────────

function _macrokernel!(kernel::SIMDKernel{W,MR,NR,Complex{TR}},
                       C::AbstractMatrix{Complex{TR}},
                       Apack::Vector{TR}, Bpack::Vector{TR},
                       ic::Int, jc::Int, mc::Int, nc::Int, kc::Int,
                       α::Complex{TR}) where {W,MR,NR,TR<:Real}
    @inbounds for jr in 0:NR:nc-1
        nr_ = min(NR, nc - jr)
        bo  = jr * 2 * kc                   # 2× for complex
        for ir in 0:MR:mc-1
            mr_ = min(MR, mc - ir)
            ao  = ir * 2 * kc
            ci  = ic + ir
            cj  = jc + jr
            if mr_ == MR && nr_ == NR
                _kernel!(kernel, C, Apack, Bpack, ao, bo, kc, ci, cj, MR, NR, α, Val(true))
            else
                _kernel!(kernel, C, Apack, Bpack, ao, bo, kc, ci, cj, mr_, nr_, α, Val(false))
            end
        end
    end
    return nothing
end

# ─── Workspace + entry point ──────────────────────────────────────────────

function gemm_workspace(::Type{Complex{TR}},
                        kernel::SIMDKernel{W,MR,NR,Complex{TR}}) where {W,MR,NR,TR<:Real}
    caches = _cache_sizes()
    MC_ = mc_block(kernel, caches)
    KC_ = kc_block(kernel, caches)
    NC_ = nc_block(kernel, caches)
    Apack = Vector{TR}(undef, 2 * cld(MC_, MR) * MR * KC_)
    Bpack = Vector{TR}(undef, 2 * cld(NC_, NR) * NR * KC_)
    return (Apack, Bpack)
end

function gemm!(C::AbstractMatrix{Complex{TR}}, A::AbstractMatrix{Complex{TR}}, B::AbstractMatrix{Complex{TR}},
               α = true, β = false;
               kernel::SIMDKernel{W,MR,NR,Complex{TR}} = default_kernel(Complex{TR}),
               Apack::Union{Vector{TR},Nothing} = nothing,
               Bpack::Union{Vector{TR},Nothing} = nothing) where {W,MR,NR,TR<:Real}
    M, N = size(C)
    size(A, 1) == M           || throw(DimensionMismatch("size(A,1) ≠ size(C,1)"))
    size(B, 2) == N           || throw(DimensionMismatch("size(B,2) ≠ size(C,2)"))
    size(A, 2) == size(B, 1)  || throw(DimensionMismatch("size(A,2) ≠ size(B,1)"))
    K = size(A, 2)

    _scale!(C, β)
    (M == 0 || N == 0 || K == 0) && return C

    caches = _cache_sizes()
    MC_ = mc_block(kernel, caches)
    KC_ = kc_block(kernel, caches)
    NC_ = nc_block(kernel, caches)

    mc_max = min(MC_, M); kc_max = min(KC_, K); nc_max = min(NC_, N)
    apack_sz = 2 * cld(mc_max, MR) * MR * kc_max
    bpack_sz = 2 * cld(nc_max, NR) * NR * kc_max
    Apack === nothing && (Apack = Vector{TR}(undef, apack_sz))
    Bpack === nothing && (Bpack = Vector{TR}(undef, bpack_sz))
    length(Apack) >= apack_sz ||
        throw(ArgumentError("Apack too small: have $(length(Apack)), need $apack_sz"))
    length(Bpack) >= bpack_sz ||
        throw(ArgumentError("Bpack too small: have $(length(Bpack)), need $bpack_sz"))
    αT = convert(Complex{TR}, α)

    @inbounds for jc in 1:NC_:N
        nc = min(NC_, N - jc + 1)
        for pc in 1:KC_:K
            kc = min(KC_, K - pc + 1)
            _pack_B!(Bpack, B, pc, jc, kc, nc, Val(NR))
            for ic in 1:MC_:M
                mc = min(MC_, M - ic + 1)
                _pack_A!(Apack, A, ic, pc, mc, kc, Val(MR))
                _macrokernel!(kernel, C, Apack, Bpack, ic, jc, mc, nc, kc, αT)
            end
        end
    end
    return C
end
