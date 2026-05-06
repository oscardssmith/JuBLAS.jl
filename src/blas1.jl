# Pointer-based BLAS Level 1 kernels.
#
# Every routine takes raw pointers + length + increments, mirroring the
# Fortran ABI exactly. Unit-stride paths use @inbounds @simd for the
# memory-bandwidth peak; non-unit/negative-stride paths walk an integer
# offset per the reference-BLAS convention (start at (n-1)*|inc| when
# inc<0, increment by inc each step).

# LP64. Shared with blas_interface.jl, defined here because this file is
# included first.
const BlasInt = Cint
#
# Reductions follow reference behaviour for invalid args:
#   nrm2/asum/dot/dotc/dotu  -> 0  if n <= 0
#   iamax                    -> 0  if n < 1 or incx <= 0  (1-indexed otherwise)
#   scal                     -> no-op if n <= 0 or incx <= 0
# axpy/copy/swap accept any sign of inc (same as reference).

@inline _start_off(n::Int, inc::Int) = inc < 0 ? (n - 1) * (-inc) : 0

# ─── axpy: y := α·x + y ────────────────────────────────────────────────────

@inline function _axpy_impl!(n::Int, α, x::Ptr{T}, incx::Int,
                              y::Ptr{T}, incy::Int) where {T}
    (n <= 0 || α == zero(α)) && return nothing
    if incx == 1 && incy == 1
        @inbounds @simd for i in 1:n
            unsafe_store!(y, unsafe_load(y, i) + α * unsafe_load(x, i), i)
        end
    else
        ix = _start_off(n, incx)
        iy = _start_off(n, incy)
        @inbounds for _ in 1:n
            unsafe_store!(y, unsafe_load(y, iy + 1) + α * unsafe_load(x, ix + 1), iy + 1)
            ix += incx
            iy += incy
        end
    end
    return nothing
end

# ─── scal: x := α·x ────────────────────────────────────────────────────────

@inline function _scal_impl!(n::Int, α, x::Ptr{T}, incx::Int) where {T}
    (n <= 0 || incx <= 0) && return nothing
    if incx == 1
        @inbounds @simd for i in 1:n
            unsafe_store!(x, α * unsafe_load(x, i), i)
        end
    else
        ix = 0
        @inbounds for _ in 1:n
            unsafe_store!(x, α * unsafe_load(x, ix + 1), ix + 1)
            ix += incx
        end
    end
    return nothing
end

# ─── copy: y := x ──────────────────────────────────────────────────────────

@inline function _copy_impl!(n::Int, x::Ptr{T}, incx::Int,
                              y::Ptr{T}, incy::Int) where {T}
    n <= 0 && return nothing
    if incx == 1 && incy == 1
        @inbounds @simd for i in 1:n
            unsafe_store!(y, unsafe_load(x, i), i)
        end
    else
        ix = _start_off(n, incx)
        iy = _start_off(n, incy)
        @inbounds for _ in 1:n
            unsafe_store!(y, unsafe_load(x, ix + 1), iy + 1)
            ix += incx
            iy += incy
        end
    end
    return nothing
end

# ─── swap: x ↔ y ───────────────────────────────────────────────────────────

@inline function _swap_impl!(n::Int, x::Ptr{T}, incx::Int,
                              y::Ptr{T}, incy::Int) where {T}
    n <= 0 && return nothing
    if incx == 1 && incy == 1
        @inbounds @simd for i in 1:n
            xi = unsafe_load(x, i)
            unsafe_store!(x, unsafe_load(y, i), i)
            unsafe_store!(y, xi, i)
        end
    else
        ix = _start_off(n, incx)
        iy = _start_off(n, incy)
        @inbounds for _ in 1:n
            xi = unsafe_load(x, ix + 1)
            unsafe_store!(x, unsafe_load(y, iy + 1), ix + 1)
            unsafe_store!(y, xi, iy + 1)
            ix += incx
            iy += incy
        end
    end
    return nothing
end

# ─── dot: real, complex (conjugated and unconjugated) ─────────────────────
#
# `_dot_impl` handles real and complex-unconjugated (sdot/ddot/cdotu/zdotu).
# `_dotc_impl` handles complex-conjugated (cdotc/zdotc) — `conj(x) * y`.

@inline function _dot_impl(::Type{Tacc}, n::Int,
                            x::Ptr{T}, incx::Int,
                            y::Ptr{T}, incy::Int) where {Tacc, T}
    s = zero(Tacc)
    n <= 0 && return s
    if incx == 1 && incy == 1
        @inbounds @simd for i in 1:n
            s += Tacc(unsafe_load(x, i)) * Tacc(unsafe_load(y, i))
        end
    else
        ix = _start_off(n, incx)
        iy = _start_off(n, incy)
        @inbounds for _ in 1:n
            s += Tacc(unsafe_load(x, ix + 1)) * Tacc(unsafe_load(y, iy + 1))
            ix += incx
            iy += incy
        end
    end
    return s
end

@inline function _dotc_impl(n::Int,
                             x::Ptr{T}, incx::Int,
                             y::Ptr{T}, incy::Int) where {T<:Complex}
    s = zero(T)
    n <= 0 && return s
    if incx == 1 && incy == 1
        @inbounds @simd for i in 1:n
            s += conj(unsafe_load(x, i)) * unsafe_load(y, i)
        end
    else
        ix = _start_off(n, incx)
        iy = _start_off(n, incy)
        @inbounds for _ in 1:n
            s += conj(unsafe_load(x, ix + 1)) * unsafe_load(y, iy + 1)
            ix += incx
            iy += incy
        end
    end
    return s
end

# `sdsdot`: real Float32 dot with Float64 accumulator and a Float32 bias.
@inline function _sdsdot_impl(n::Int, sb::Float32,
                               x::Ptr{Float32}, incx::Int,
                               y::Ptr{Float32}, incy::Int)
    s = Float64(sb)
    n <= 0 && return Float32(s)
    s += _dot_impl(Float64, n, x, incx, y, incy)
    return Float32(s)
end

# ─── nrm2: scaled-sum-of-squares (overflow-safe) ──────────────────────────

@inline function _nrm2_real_impl(n::Int, x::Ptr{T}, incx::Int) where {T<:Real}
    (n <= 0 || incx <= 0) && return zero(T)
    n == 1 && return abs(unsafe_load(x, 1))
    scale = zero(T)
    ssq   = one(T)
    ix = 0
    @inbounds for _ in 1:n
        v = unsafe_load(x, ix + 1)
        if v != zero(T)
            absv = abs(v)
            if scale < absv
                ssq = one(T) + ssq * (scale / absv)^2
                scale = absv
            else
                ssq += (absv / scale)^2
            end
        end
        ix += incx
    end
    return scale * sqrt(ssq)
end

@inline function _nrm2_complex_impl(n::Int, x::Ptr{Complex{TR}}, incx::Int) where {TR<:Real}
    (n <= 0 || incx <= 0) && return zero(TR)
    scale = zero(TR)
    ssq   = one(TR)
    ix = 0
    @inbounds for _ in 1:n
        v = unsafe_load(x, ix + 1)
        for component in (real(v), imag(v))
            if component != zero(TR)
                absc = abs(component)
                if scale < absc
                    ssq = one(TR) + ssq * (scale / absc)^2
                    scale = absc
                else
                    ssq += (absc / scale)^2
                end
            end
        end
        ix += incx
    end
    return scale * sqrt(ssq)
end

# ─── asum: Σ|x[i]|  (complex: Σ(|Re|+|Im|), per reference BLAS) ────────────

@inline function _asum_real_impl(n::Int, x::Ptr{T}, incx::Int) where {T<:Real}
    s = zero(T)
    (n <= 0 || incx <= 0) && return s
    if incx == 1
        @inbounds @simd for i in 1:n
            s += abs(unsafe_load(x, i))
        end
    else
        ix = 0
        @inbounds for _ in 1:n
            s += abs(unsafe_load(x, ix + 1))
            ix += incx
        end
    end
    return s
end

@inline function _asum_complex_impl(n::Int, x::Ptr{Complex{TR}}, incx::Int) where {TR<:Real}
    s = zero(TR)
    (n <= 0 || incx <= 0) && return s
    if incx == 1
        @inbounds @simd for i in 1:n
            v = unsafe_load(x, i)
            s += abs(real(v)) + abs(imag(v))
        end
    else
        ix = 0
        @inbounds for _ in 1:n
            v = unsafe_load(x, ix + 1)
            s += abs(real(v)) + abs(imag(v))
            ix += incx
        end
    end
    return s
end

# ─── iamax: 1-indexed argmax of |x[i]| (complex: |Re|+|Im|) ────────────────

@inline function _iamax_real_impl(n::Int, x::Ptr{T}, incx::Int) where {T<:Real}
    (n < 1 || incx <= 0) && return BlasInt(0)
    n == 1 && return BlasInt(1)
    best_i = 1
    best_v = abs(unsafe_load(x, 1))
    ix = incx
    @inbounds for i in 2:n
        v = abs(unsafe_load(x, ix + 1))
        if v > best_v
            best_v = v
            best_i = i
        end
        ix += incx
    end
    return BlasInt(best_i)
end

@inline function _iamax_complex_impl(n::Int, x::Ptr{Complex{TR}}, incx::Int) where {TR<:Real}
    (n < 1 || incx <= 0) && return BlasInt(0)
    v0 = unsafe_load(x, 1)
    n == 1 && return BlasInt(1)
    best_i = 1
    best_v = abs(real(v0)) + abs(imag(v0))
    ix = incx
    @inbounds for i in 2:n
        v = unsafe_load(x, ix + 1)
        cand = abs(real(v)) + abs(imag(v))
        if cand > best_v
            best_v = cand
            best_i = i
        end
        ix += incx
    end
    return BlasInt(best_i)
end

# ─── rot: apply Givens (or Jacobi-style) rotation ─────────────────────────
#
# [x;y] := [c s; -s c] * [x;y]. Used by srot/drot (real) and csrot/zdrot
# (real c,s applied to complex x,y) — the same arithmetic works for both
# because Julia's `Float * Complex` promotes naturally.

@inline function _rot_impl!(n::Int,
                             x::Ptr{Tv}, incx::Int,
                             y::Ptr{Tv}, incy::Int,
                             c::Tc, s::Tc) where {Tv, Tc}
    n <= 0 && return nothing
    if incx == 1 && incy == 1
        @inbounds @simd for i in 1:n
            xi = unsafe_load(x, i)
            yi = unsafe_load(y, i)
            unsafe_store!(x,  c*xi + s*yi, i)
            unsafe_store!(y, -s*xi + c*yi, i)
        end
    else
        ix = _start_off(n, incx)
        iy = _start_off(n, incy)
        @inbounds for _ in 1:n
            xi = unsafe_load(x, ix + 1)
            yi = unsafe_load(y, iy + 1)
            unsafe_store!(x,  c*xi + s*yi, ix + 1)
            unsafe_store!(y, -s*xi + c*yi, iy + 1)
            ix += incx
            iy += incy
        end
    end
    return nothing
end

# ─── rotg (real): generate Givens that zeros b ────────────────────────────
#
# On entry: a, b. On exit: a := r, b := z (a recovery scalar), c := cosθ, s := sinθ.
# Reference algorithm (BLAS DROTG): pick `roe` to break sign ties, scale to
# avoid overflow when computing r = ±√(a² + b²).

@inline function _rotg_impl!(ap::Ptr{T}, bp::Ptr{T},
                              cp::Ptr{T}, sp::Ptr{T}) where {T<:Real}
    a = unsafe_load(ap)
    b = unsafe_load(bp)
    roe = abs(a) > abs(b) ? a : b
    scale = abs(a) + abs(b)
    if scale == zero(T)
        unsafe_store!(cp, one(T))
        unsafe_store!(sp, zero(T))
        unsafe_store!(ap, zero(T))
        unsafe_store!(bp, zero(T))
    else
        as = a / scale
        bs = b / scale
        r = scale * sqrt(as*as + bs*bs)
        # BLAS uses SIGN(1, roe): +1 if roe ≥ 0 else -1. (roe is non-zero
        # here since scale > 0.)
        r = (roe >= zero(T) ? r : -r)
        c = a / r
        s = b / r
        z = one(T)
        if abs(a) > abs(b)
            z = s
        elseif c != zero(T)
            z = one(T) / c
        end
        unsafe_store!(cp, c)
        unsafe_store!(sp, s)
        unsafe_store!(ap, r)
        unsafe_store!(bp, z)
    end
    return nothing
end

# ─── rotg (complex): generate Givens for complex a,b ──────────────────────
#
# Result has c real and s complex. Reference algorithm (BLAS CROTG/ZROTG):
# special-case |a|=0 (set c=0, s=(1,0), a:=b), otherwise compute via the
# scaled norm to avoid overflow.

@inline function _rotg_complex_impl!(ap::Ptr{Complex{TR}}, bp::Ptr{Complex{TR}},
                                      cp::Ptr{TR}, sp::Ptr{Complex{TR}}) where {TR<:Real}
    a = unsafe_load(ap)
    b = unsafe_load(bp)
    abs_a = abs(a)
    if abs_a == zero(TR)
        unsafe_store!(cp, zero(TR))
        unsafe_store!(sp, complex(one(TR), zero(TR)))
        unsafe_store!(ap, b)
        # b unchanged
    else
        scale = abs_a + abs(b)
        as = a / scale
        bs = b / scale
        norm = scale * sqrt(abs2(as) + abs2(bs))
        alpha = a / abs_a       # unit complex in direction of a
        unsafe_store!(cp, abs_a / norm)
        unsafe_store!(sp, alpha * conj(b) / norm)
        unsafe_store!(ap, alpha * norm)
        # b unchanged
    end
    return nothing
end

# ─── rotm: apply modified Givens (5-element param array) ──────────────────
#
# param[1] is a flag selecting which 2×2 form H takes:
#   -2: H = I               (no-op)
#   -1: H = [h11 h12; h21 h22] (full)
#    0: H = [1   h12; h21 1  ]
#    1: H = [h11 1  ; -1  h22]
# h11/h21/h12/h22 read from param[2..5] (1-indexed: 2,3,4,5).

@inline function _rotm_impl!(n::Int,
                              x::Ptr{T}, incx::Int,
                              y::Ptr{T}, incy::Int,
                              param::Ptr{T}) where {T<:Real}
    n <= 0 && return nothing
    flag = unsafe_load(param, 1)
    flag == T(-2) && return nothing

    if flag == T(-1)
        h11 = unsafe_load(param, 2)
        h21 = unsafe_load(param, 3)
        h12 = unsafe_load(param, 4)
        h22 = unsafe_load(param, 5)
    elseif flag == zero(T)
        h11 = one(T)
        h21 = unsafe_load(param, 3)
        h12 = unsafe_load(param, 4)
        h22 = one(T)
    elseif flag == one(T)
        h11 = unsafe_load(param, 2)
        h21 = -one(T)
        h12 = one(T)
        h22 = unsafe_load(param, 5)
    else
        return nothing  # invalid flag
    end

    if incx == 1 && incy == 1
        @inbounds @simd for i in 1:n
            w = unsafe_load(x, i)
            z = unsafe_load(y, i)
            unsafe_store!(x, w*h11 + z*h12, i)
            unsafe_store!(y, w*h21 + z*h22, i)
        end
    else
        ix = _start_off(n, incx)
        iy = _start_off(n, incy)
        @inbounds for _ in 1:n
            w = unsafe_load(x, ix + 1)
            z = unsafe_load(y, iy + 1)
            unsafe_store!(x, w*h11 + z*h12, ix + 1)
            unsafe_store!(y, w*h21 + z*h22, iy + 1)
            ix += incx
            iy += incy
        end
    end
    return nothing
end

# ─── rotmg: generate modified Givens ──────────────────────────────────────
#
# Direct transcription of reference BLAS DROTMG / SROTMG. Constants
# (GAM = 2¹², GAMSQ = 2²⁴, RGAMSQ = 2⁻²⁴) are the same in both precisions.
# Updates d1, d2, x1 in place; y1 unchanged. param[1..5] receives the H
# encoding; only the entries relevant to the chosen flag are written.

@inline function _rotmg_impl!(d1p::Ptr{T}, d2p::Ptr{T},
                               x1p::Ptr{T}, y1p::Ptr{T},
                               param::Ptr{T}) where {T<:Real}
    GAM    = T(4096)
    GAMSQ  = T(16777216)
    RGAMSQ = T(5.9604645e-8)

    d1 = unsafe_load(d1p)
    d2 = unsafe_load(d2p)
    x1 = unsafe_load(x1p)
    y1 = unsafe_load(y1p)

    flag = zero(T)
    h11 = zero(T); h12 = zero(T); h21 = zero(T); h22 = zero(T)

    if d1 < zero(T)
        flag = -one(T)
        d1 = zero(T); d2 = zero(T); x1 = zero(T)
    else
        p2 = d2 * y1
        if p2 == zero(T)
            unsafe_store!(param, T(-2), 1)
            return nothing
        end
        p1 = d1 * x1
        q2 = p2 * y1
        q1 = p1 * x1
        if abs(q1) > abs(q2)
            h21 = -y1 / x1
            h12 = p2 / p1
            u = one(T) - h12 * h21
            if u > zero(T)
                flag = zero(T)
                d1 = d1 / u
                d2 = d2 / u
                x1 = x1 * u
            end
        else
            if q2 < zero(T)
                flag = -one(T)
                h11 = zero(T); h12 = zero(T); h21 = zero(T); h22 = zero(T)
                d1 = zero(T); d2 = zero(T); x1 = zero(T)
            else
                flag = one(T)
                h11 = p1 / p2
                h22 = x1 / y1
                u = one(T) + h11 * h22
                tmp = d2 / u
                d2 = d1 / u
                d1 = tmp
                x1 = y1 * u
            end
        end

        # Scale-back loops to keep d1, d2 in [RGAMSQ, GAMSQ].
        if d1 != zero(T)
            while d1 <= RGAMSQ || d1 >= GAMSQ
                if flag == zero(T)
                    h11 = one(T); h22 = one(T)
                    flag = -one(T)
                else
                    h21 = -one(T); h12 = one(T)
                    flag = -one(T)
                end
                if d1 <= RGAMSQ
                    d1 = d1 * GAM * GAM
                    x1 = x1 / GAM
                    h11 = h11 / GAM
                    h12 = h12 / GAM
                else
                    d1 = d1 / (GAM * GAM)
                    x1 = x1 * GAM
                    h11 = h11 * GAM
                    h12 = h12 * GAM
                end
            end
        end

        if d2 != zero(T)
            while abs(d2) <= RGAMSQ || abs(d2) >= GAMSQ
                if flag == zero(T)
                    h11 = one(T); h22 = one(T)
                    flag = -one(T)
                else
                    h21 = -one(T); h12 = one(T)
                    flag = -one(T)
                end
                if abs(d2) <= RGAMSQ
                    d2 = d2 * GAM * GAM
                    h21 = h21 / GAM
                    h22 = h22 / GAM
                else
                    d2 = d2 / (GAM * GAM)
                    h21 = h21 * GAM
                    h22 = h22 * GAM
                end
            end
        end
    end

    # Write back only the entries the chosen flag uses.
    if flag < zero(T)
        unsafe_store!(param, h11, 2)
        unsafe_store!(param, h21, 3)
        unsafe_store!(param, h12, 4)
        unsafe_store!(param, h22, 5)
    elseif flag == zero(T)
        unsafe_store!(param, h21, 3)
        unsafe_store!(param, h12, 4)
    else
        unsafe_store!(param, h11, 2)
        unsafe_store!(param, h22, 5)
    end
    unsafe_store!(param, flag, 1)

    unsafe_store!(d1p, d1)
    unsafe_store!(d2p, d2)
    unsafe_store!(x1p, x1)
    return nothing
end
