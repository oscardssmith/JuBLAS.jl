# Pointer-based BLAS Level 2 kernels — general matrix-vector core.
#
# Covers ?gemv (matrix-vector multiply with N/T/C trans) and the rank-1
# update family (?ger / ?geru / ?gerc). Banded / packed / triangular /
# symmetric / Hermitian forms come in a later pass.
#
# Same conventions as blas1.jl: pointers + sizes + increments, no
# AbstractArray wrapping. All the vectors honour the standard BLAS
# stride convention (start at (n-1)*|inc| when inc<0, increment by inc
# each step). Matrix `A` is column-major with leading dimension `lda`,
# so element (i,j) is at byte offset `((j-1)*lda + (i-1)) * sizeof(T)`,
# i.e. `unsafe_load(A, i + (j-1)*lda)` in Julia's 1-indexed pointer API.

# y := β·y. Honours β=0 by overwriting (so caller's y can hold NaN).
@inline function _scale_y!(y::Ptr{T}, n::Int, incy::Int, β) where {T}
    n <= 0 && return nothing
    if β == zero(β)
        if incy == 1
            @inbounds @simd for i in 1:n
                unsafe_store!(y, zero(T), i)
            end
        else
            iy = _start_off(n, incy)
            @inbounds for _ in 1:n
                unsafe_store!(y, zero(T), iy + 1)
                iy += incy
            end
        end
    elseif β != one(β)
        if incy == 1
            @inbounds @simd for i in 1:n
                unsafe_store!(y, β * unsafe_load(y, i), i)
            end
        else
            iy = _start_off(n, incy)
            @inbounds for _ in 1:n
                unsafe_store!(y, β * unsafe_load(y, iy + 1), iy + 1)
                iy += incy
            end
        end
    end
    return nothing
end

# ─── gemv: y := α·op(A)·x + β·y ───────────────────────────────────────────
#
# The 'N' branch loops over columns of A: for each j, it reads x[j] once
# and AXPYs column j into y. This gives unit-stride loads on A (the inner
# loop walks down a column) and unit-stride writes to y when incy=1, hitting
# the memory-bandwidth peak the user expects from L2. The 'T' / 'C' branches
# loop over output rows j of op(A) and accumulate a scalar dot product down
# A's column j; same column-walking access pattern, just folded into a sum.

@inline function _gemv_impl!(trans::UInt8, m::Int, n::Int,
                              α, A::Ptr{T}, lda::Int,
                              x::Ptr{T}, incx::Int,
                              β, y::Ptr{T}, incy::Int) where {T}
    (m <= 0 || n <= 0) && return nothing
    if _isN(trans)
        # y is length m, x is length n.
        _scale_y!(y, m, incy, β)
        α == zero(α) && return nothing
        ix = _start_off(n, incx)
        @inbounds for j in 1:n
            tmp = α * unsafe_load(x, ix + 1)
            colj = (j - 1) * lda
            iy = _start_off(m, incy)
            if incy == 1
                @simd for i in 1:m
                    unsafe_store!(y, unsafe_load(y, i) + tmp * unsafe_load(A, i + colj), i)
                end
            else
                for i in 1:m
                    unsafe_store!(y, unsafe_load(y, iy + 1) + tmp * unsafe_load(A, i + colj), iy + 1)
                    iy += incy
                end
            end
            ix += incx
        end
    elseif _isT(trans)
        # y is length n, x is length m. y[j] += α · ⟨A[:,j], x⟩
        _scale_y!(y, n, incy, β)
        α == zero(α) && return nothing
        iy = _start_off(n, incy)
        @inbounds for j in 1:n
            colj = (j - 1) * lda
            s = zero(T)
            ix = _start_off(m, incx)
            if incx == 1
                @simd for i in 1:m
                    s += unsafe_load(A, i + colj) * unsafe_load(x, i)
                end
            else
                for i in 1:m
                    s += unsafe_load(A, i + colj) * unsafe_load(x, ix + 1)
                    ix += incx
                end
            end
            unsafe_store!(y, unsafe_load(y, iy + 1) + α * s, iy + 1)
            iy += incy
        end
    else
        # 'C' / adjoint: y[j] += α · ⟨conj(A[:,j]), x⟩ . For T<:Real conj is
        # identity and the compiler folds it away — keeping the unified
        # branch costs nothing and saves a duplicate copy of the loop.
        _scale_y!(y, n, incy, β)
        α == zero(α) && return nothing
        iy = _start_off(n, incy)
        @inbounds for j in 1:n
            colj = (j - 1) * lda
            s = zero(T)
            ix = _start_off(m, incx)
            if incx == 1
                @simd for i in 1:m
                    s += conj(unsafe_load(A, i + colj)) * unsafe_load(x, i)
                end
            else
                for i in 1:m
                    s += conj(unsafe_load(A, i + colj)) * unsafe_load(x, ix + 1)
                    ix += incx
                end
            end
            unsafe_store!(y, unsafe_load(y, iy + 1) + α * s, iy + 1)
            iy += incy
        end
    end
    return nothing
end

# ─── ger / geru / gerc: A := α·x·op(y) + A ────────────────────────────────
#
# ger / geru: op(y) = yᵀ            (no conjugation)
# gerc:       op(y) = yᴴ = conj(y)ᵀ
#
# Loop order j-outer, i-inner — column-walking writes into A and unit-stride
# loads of x on the fast path.

@inline function _geru_impl!(m::Int, n::Int, α,
                              x::Ptr{T}, incx::Int,
                              y::Ptr{T}, incy::Int,
                              A::Ptr{T}, lda::Int) where {T}
    (m <= 0 || n <= 0 || α == zero(α)) && return nothing
    iy = _start_off(n, incy)
    @inbounds for j in 1:n
        αyj = α * unsafe_load(y, iy + 1)
        colj = (j - 1) * lda
        ix = _start_off(m, incx)
        if incx == 1
            @simd for i in 1:m
                off = i + colj
                unsafe_store!(A, unsafe_load(A, off) + αyj * unsafe_load(x, i), off)
            end
        else
            for i in 1:m
                off = i + colj
                unsafe_store!(A, unsafe_load(A, off) + αyj * unsafe_load(x, ix + 1), off)
                ix += incx
            end
        end
        iy += incy
    end
    return nothing
end

@inline function _gerc_impl!(m::Int, n::Int, α,
                              x::Ptr{T}, incx::Int,
                              y::Ptr{T}, incy::Int,
                              A::Ptr{T}, lda::Int) where {T}
    (m <= 0 || n <= 0 || α == zero(α)) && return nothing
    iy = _start_off(n, incy)
    @inbounds for j in 1:n
        αyj = α * conj(unsafe_load(y, iy + 1))
        colj = (j - 1) * lda
        ix = _start_off(m, incx)
        if incx == 1
            @simd for i in 1:m
                off = i + colj
                unsafe_store!(A, unsafe_load(A, off) + αyj * unsafe_load(x, i), off)
            end
        else
            for i in 1:m
                off = i + colj
                unsafe_store!(A, unsafe_load(A, off) + αyj * unsafe_load(x, ix + 1), off)
                ix += incx
            end
        end
        iy += incy
    end
    return nothing
end

# ─── uplo helpers ─────────────────────────────────────────────────────────

@inline _isU(c::UInt8) = (c == UInt8('U')) | (c == UInt8('u'))

# ─── symv: y := α·A·x + β·y    (A symmetric, real eltype) ──────────────────
#
# Reference BLAS DSYMV: walk j-outer; for each column j of the stored
# triangle, fold the "row j" half (mirrored from the unstored triangle)
# into a tmp2 accumulator and the "column j above/below diag" half directly
# into y. Same arithmetic count as a full gemv, but only n(n+1)/2 reads
# from A (the stored triangle).

@inline function _symv_impl!(uplo::UInt8, n::Int, α,
                              A::Ptr{T}, lda::Int,
                              x::Ptr{T}, incx::Int,
                              β, y::Ptr{T}, incy::Int) where {T<:Real}
    n <= 0 && return nothing
    _scale_y!(y, n, incy, β)
    α == zero(α) && return nothing
    kx0 = _start_off(n, incx)
    ky0 = _start_off(n, incy)
    if _isU(uplo)
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            colj = (j - 1) * lda
            ix = kx0; iy = ky0
            for i in 1:j-1
                aij = unsafe_load(A, i + colj)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += aij * unsafe_load(x, ix + 1)
                ix += incx; iy += incy
            end
            ajj = unsafe_load(A, j + colj)
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj + α * tmp2, jy + 1)
        end
    else  # lower
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            colj = (j - 1) * lda
            ajj = unsafe_load(A, j + colj)
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj, jy + 1)
            ix = kx0 + j * incx
            iy = ky0 + j * incy
            for i in j+1:n
                aij = unsafe_load(A, i + colj)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += aij * unsafe_load(x, ix + 1)
                ix += incx; iy += incy
            end
            unsafe_store!(y, unsafe_load(y, jy + 1) + α * tmp2, jy + 1)
        end
    end
    return nothing
end

# ─── hemv: y := α·A·x + β·y    (A Hermitian, complex eltype) ───────────────
#
# Same loop shape as symv, except the mirrored-triangle contribution
# conjugates A and the diagonal element is treated as real (per BLAS spec).

@inline function _hemv_impl!(uplo::UInt8, n::Int, α,
                              A::Ptr{Complex{TR}}, lda::Int,
                              x::Ptr{Complex{TR}}, incx::Int,
                              β, y::Ptr{Complex{TR}}, incy::Int) where {TR<:Real}
    T = Complex{TR}
    n <= 0 && return nothing
    _scale_y!(y, n, incy, β)
    α == zero(α) && return nothing
    kx0 = _start_off(n, incx)
    ky0 = _start_off(n, incy)
    if _isU(uplo)
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            colj = (j - 1) * lda
            ix = kx0; iy = ky0
            for i in 1:j-1
                aij = unsafe_load(A, i + colj)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += conj(aij) * unsafe_load(x, ix + 1)
                ix += incx; iy += incy
            end
            ajj_re = real(unsafe_load(A, j + colj))
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj_re + α * tmp2, jy + 1)
        end
    else
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            colj = (j - 1) * lda
            ajj_re = real(unsafe_load(A, j + colj))
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj_re, jy + 1)
            ix = kx0 + j * incx
            iy = ky0 + j * incy
            for i in j+1:n
                aij = unsafe_load(A, i + colj)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += conj(aij) * unsafe_load(x, ix + 1)
                ix += incx; iy += incy
            end
            unsafe_store!(y, unsafe_load(y, jy + 1) + α * tmp2, jy + 1)
        end
    end
    return nothing
end

# ─── syr: A := α·x·xᵀ + A    (A symmetric, real eltype) ───────────────────

@inline function _syr_impl!(uplo::UInt8, n::Int, α,
                             x::Ptr{T}, incx::Int,
                             A::Ptr{T}, lda::Int) where {T<:Real}
    (n <= 0 || α == zero(α)) && return nothing
    kx0 = _start_off(n, incx)
    if _isU(uplo)
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            xj = unsafe_load(x, jx + 1)
            xj == zero(T) && continue
            tmp = α * xj
            colj = (j - 1) * lda
            ix = kx0
            for i in 1:j
                off = i + colj
                unsafe_store!(A, unsafe_load(A, off) + tmp * unsafe_load(x, ix + 1), off)
                ix += incx
            end
        end
    else
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            xj = unsafe_load(x, jx + 1)
            xj == zero(T) && continue
            tmp = α * xj
            colj = (j - 1) * lda
            ix = jx
            for i in j:n
                off = i + colj
                unsafe_store!(A, unsafe_load(A, off) + tmp * unsafe_load(x, ix + 1), off)
                ix += incx
            end
        end
    end
    return nothing
end

# ─── syr2: A := α·x·yᵀ + α·y·xᵀ + A    (real symmetric) ──────────────────

@inline function _syr2_impl!(uplo::UInt8, n::Int, α,
                              x::Ptr{T}, incx::Int,
                              y::Ptr{T}, incy::Int,
                              A::Ptr{T}, lda::Int) where {T<:Real}
    (n <= 0 || α == zero(α)) && return nothing
    kx0 = _start_off(n, incx)
    ky0 = _start_off(n, incy)
    if _isU(uplo)
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            xj = unsafe_load(x, jx + 1)
            yj = unsafe_load(y, jy + 1)
            (xj == zero(T) && yj == zero(T)) && continue
            tmp1 = α * yj
            tmp2 = α * xj
            colj = (j - 1) * lda
            ix = kx0; iy = ky0
            for i in 1:j
                off = i + colj
                xi = unsafe_load(x, ix + 1)
                yi = unsafe_load(y, iy + 1)
                unsafe_store!(A, unsafe_load(A, off) + xi * tmp1 + yi * tmp2, off)
                ix += incx; iy += incy
            end
        end
    else
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            xj = unsafe_load(x, jx + 1)
            yj = unsafe_load(y, jy + 1)
            (xj == zero(T) && yj == zero(T)) && continue
            tmp1 = α * yj
            tmp2 = α * xj
            colj = (j - 1) * lda
            ix = jx; iy = jy
            for i in j:n
                off = i + colj
                xi = unsafe_load(x, ix + 1)
                yi = unsafe_load(y, iy + 1)
                unsafe_store!(A, unsafe_load(A, off) + xi * tmp1 + yi * tmp2, off)
                ix += incx; iy += incy
            end
        end
    end
    return nothing
end

# ─── her: A := α·x·xᴴ + A    (A Hermitian, α real) ────────────────────────
#
# Diagonal element keeps its real-valued invariant: write only the real
# part of the rank-1 contribution there.

@inline function _her_impl!(uplo::UInt8, n::Int, α::TR,
                             x::Ptr{Complex{TR}}, incx::Int,
                             A::Ptr{Complex{TR}}, lda::Int) where {TR<:Real}
    T = Complex{TR}
    (n <= 0 || α == zero(α)) && return nothing
    kx0 = _start_off(n, incx)
    if _isU(uplo)
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            xj = unsafe_load(x, jx + 1)
            xj == zero(T) && continue
            tmp = α * conj(xj)
            colj = (j - 1) * lda
            ix = kx0
            for i in 1:j-1
                off = i + colj
                unsafe_store!(A, unsafe_load(A, off) + unsafe_load(x, ix + 1) * tmp, off)
                ix += incx
            end
            ajj = unsafe_load(A, j + colj)
            unsafe_store!(A, complex(real(ajj) + real(xj * tmp), zero(TR)), j + colj)
        end
    else
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            xj = unsafe_load(x, jx + 1)
            xj == zero(T) && continue
            tmp = α * conj(xj)
            colj = (j - 1) * lda
            ajj = unsafe_load(A, j + colj)
            unsafe_store!(A, complex(real(ajj) + real(xj * tmp), zero(TR)), j + colj)
            ix = kx0 + j * incx
            for i in j+1:n
                off = i + colj
                unsafe_store!(A, unsafe_load(A, off) + unsafe_load(x, ix + 1) * tmp, off)
                ix += incx
            end
        end
    end
    return nothing
end

# ─── trmv: x := op(A)·x    (A triangular, in place) ──────────────────────
#
# 12-way dispatch over uplo × trans × diag — but the column-walking pattern
# only changes with (uplo, trans), so the body is six branches. Each branch
# checks `nounit` (= diag != 'U' / 'u') to decide whether to multiply by
# A[j,j]. For trans='C' on a complex type the diagonal is conjugated too.
#
# In-place safety: the algorithms walk columns in an order that reads
# x[j_orig] before any other column writes to it.
#   trans='N', uplo='U': j=1..n, modifies x[i] for i<j and x[j] last
#   trans='N', uplo='L': j=n..1, modifies x[i] for i>j and x[j] last
#   trans='T'/'C', uplo='U': j=n..1, x[j] = sum_{i≤j} A[i,j]·x[i]
#   trans='T'/'C', uplo='L': j=1..n, x[j] = sum_{i≥j} A[i,j]·x[i]

@inline function _trmv_impl!(uplo::UInt8, trans::UInt8, diag::UInt8,
                              n::Int, A::Ptr{T}, lda::Int,
                              x::Ptr{T}, incx::Int) where {T}
    n <= 0 && return nothing
    nounit = !_isU(diag)
    kx0 = _start_off(n, incx)

    if _isN(trans)
        if _isU(uplo)
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    colj = (j - 1) * lda
                    ix = kx0
                    for i in 1:j-1
                        unsafe_store!(x,
                            unsafe_load(x, ix + 1) + xj * unsafe_load(A, i + colj),
                            ix + 1)
                        ix += incx
                    end
                    if nounit
                        unsafe_store!(x, xj * unsafe_load(A, j + colj), jx + 1)
                    end
                end
            end
        else  # uplo = L
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    colj = (j - 1) * lda
                    ix = kx0 + j * incx
                    for i in j+1:n
                        unsafe_store!(x,
                            unsafe_load(x, ix + 1) + xj * unsafe_load(A, i + colj),
                            ix + 1)
                        ix += incx
                    end
                    if nounit
                        unsafe_store!(x, xj * unsafe_load(A, j + colj), jx + 1)
                    end
                end
            end
        end
    elseif _isT(trans)
        if _isU(uplo)
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * unsafe_load(A, j + colj)
                end
                ix = kx0
                for i in 1:j-1
                    tmp += unsafe_load(A, i + colj) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        else
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * unsafe_load(A, j + colj)
                end
                ix = kx0 + j * incx
                for i in j+1:n
                    tmp += unsafe_load(A, i + colj) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        end
    else  # trans = 'C' — conjugate transpose (identity-fold for real T)
        if _isU(uplo)
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * conj(unsafe_load(A, j + colj))
                end
                ix = kx0
                for i in 1:j-1
                    tmp += conj(unsafe_load(A, i + colj)) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        else
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * conj(unsafe_load(A, j + colj))
                end
                ix = kx0 + j * incx
                for i in j+1:n
                    tmp += conj(unsafe_load(A, i + colj)) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        end
    end
    return nothing
end

# ─── trsv: solve op(A)·x = b for x, in place ──────────────────────────────
#
# Column-walking back/forward substitution. Same six (uplo, trans) branches
# as trmv; here the diagonal step DIVIDES x[j] by A[j,j] (or skips for
# unit diag) and inner loops SUBTRACT A·x contributions.

@inline function _trsv_impl!(uplo::UInt8, trans::UInt8, diag::UInt8,
                              n::Int, A::Ptr{T}, lda::Int,
                              x::Ptr{T}, incx::Int) where {T}
    n <= 0 && return nothing
    nounit = !_isU(diag)
    kx0 = _start_off(n, incx)

    if _isN(trans)
        if _isU(uplo)
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    colj = (j - 1) * lda
                    if nounit
                        xj = xj / unsafe_load(A, j + colj)
                        unsafe_store!(x, xj, jx + 1)
                    end
                    ix = kx0
                    for i in 1:j-1
                        unsafe_store!(x,
                            unsafe_load(x, ix + 1) - xj * unsafe_load(A, i + colj),
                            ix + 1)
                        ix += incx
                    end
                end
            end
        else
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    colj = (j - 1) * lda
                    if nounit
                        xj = xj / unsafe_load(A, j + colj)
                        unsafe_store!(x, xj, jx + 1)
                    end
                    ix = kx0 + j * incx
                    for i in j+1:n
                        unsafe_store!(x,
                            unsafe_load(x, ix + 1) - xj * unsafe_load(A, i + colj),
                            ix + 1)
                        ix += incx
                    end
                end
            end
        end
    elseif _isT(trans)
        if _isU(uplo)
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                ix = kx0
                for i in 1:j-1
                    tmp -= unsafe_load(A, i + colj) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                if nounit
                    tmp = tmp / unsafe_load(A, j + colj)
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        else
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                ix = kx0 + j * incx
                for i in j+1:n
                    tmp -= unsafe_load(A, i + colj) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                if nounit
                    tmp = tmp / unsafe_load(A, j + colj)
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        end
    else  # 'C'
        if _isU(uplo)
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                ix = kx0
                for i in 1:j-1
                    tmp -= conj(unsafe_load(A, i + colj)) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                if nounit
                    tmp = tmp / conj(unsafe_load(A, j + colj))
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        else
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                ix = kx0 + j * incx
                for i in j+1:n
                    tmp -= conj(unsafe_load(A, i + colj)) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                if nounit
                    tmp = tmp / conj(unsafe_load(A, j + colj))
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        end
    end
    return nothing
end

# ─── her2: A := α·x·yᴴ + ᾱ·y·xᴴ + A    (A Hermitian, α complex) ──────────

# ─── Banded storage helpers ──────────────────────────────────────────────
#
# General band (kl subdiagonals, ku superdiagonals): A[i,j] is at
# AB[ku+1+i-j, j], i.e. pointer offset (j-1)*lda + (ku+i-j).
# Symmetric/Hermitian/triangular band, uplo='U' (k superdiagonals):
#   A[i,j] at AB[k+1+i-j, j]  for max(1,j-k) ≤ i ≤ j
# uplo='L' (k subdiagonals):
#   A[i,j] at AB[1+i-j, j]    for j ≤ i ≤ min(n,j+k)

# ─── gbmv: y := α·op(A)·x + β·y    (A general band) ──────────────────────

@inline function _gbmv_impl!(trans::UInt8, m::Int, n::Int,
                              kl::Int, ku::Int,
                              α, A::Ptr{T}, lda::Int,
                              x::Ptr{T}, incx::Int,
                              β, y::Ptr{T}, incy::Int) where {T}
    (m <= 0 || n <= 0) && return nothing
    leny = _isN(trans) ? m : n
    lenx = _isN(trans) ? n : m
    _scale_y!(y, leny, incy, β)
    α == zero(α) && return nothing
    kx0 = _start_off(lenx, incx)
    ky0 = _start_off(leny, incy)
    if _isN(trans)
        @inbounds for j in 1:n
            xj = unsafe_load(x, kx0 + (j - 1) * incx + 1)
            tmp = α * xj
            colj = (j - 1) * lda
            i_lo = max(1, j - ku)
            i_hi = min(m, j + kl)
            iy = ky0 + (i_lo - 1) * incy
            for i in i_lo:i_hi
                aij = unsafe_load(A, colj + (ku + i - j) + 1)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp * aij, iy + 1)
                iy += incy
            end
        end
    elseif _isT(trans)
        @inbounds for j in 1:n
            tmp = zero(T)
            colj = (j - 1) * lda
            i_lo = max(1, j - ku)
            i_hi = min(m, j + kl)
            ix = kx0 + (i_lo - 1) * incx
            for i in i_lo:i_hi
                tmp += unsafe_load(A, colj + (ku + i - j) + 1) * unsafe_load(x, ix + 1)
                ix += incx
            end
            jy = ky0 + (j - 1) * incy
            unsafe_store!(y, unsafe_load(y, jy + 1) + α * tmp, jy + 1)
        end
    else  # 'C'
        @inbounds for j in 1:n
            tmp = zero(T)
            colj = (j - 1) * lda
            i_lo = max(1, j - ku)
            i_hi = min(m, j + kl)
            ix = kx0 + (i_lo - 1) * incx
            for i in i_lo:i_hi
                tmp += conj(unsafe_load(A, colj + (ku + i - j) + 1)) * unsafe_load(x, ix + 1)
                ix += incx
            end
            jy = ky0 + (j - 1) * incy
            unsafe_store!(y, unsafe_load(y, jy + 1) + α * tmp, jy + 1)
        end
    end
    return nothing
end

# ─── sbmv: y := α·A·x + β·y    (A symmetric band, real) ──────────────────

@inline function _sbmv_impl!(uplo::UInt8, n::Int, k::Int, α,
                              A::Ptr{T}, lda::Int,
                              x::Ptr{T}, incx::Int,
                              β, y::Ptr{T}, incy::Int) where {T<:Real}
    n <= 0 && return nothing
    _scale_y!(y, n, incy, β)
    α == zero(α) && return nothing
    kx0 = _start_off(n, incx)
    ky0 = _start_off(n, incy)
    if _isU(uplo)
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            colj = (j - 1) * lda
            i_lo = max(1, j - k)
            iy = ky0 + (i_lo - 1) * incy
            ix = kx0 + (i_lo - 1) * incx
            for i in i_lo:j-1
                aij = unsafe_load(A, colj + (k + i - j) + 1)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += aij * unsafe_load(x, ix + 1)
                iy += incy; ix += incx
            end
            ajj = unsafe_load(A, colj + k + 1)  # diagonal
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj + α * tmp2, jy + 1)
        end
    else
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            colj = (j - 1) * lda
            ajj = unsafe_load(A, colj + 1)  # diagonal at AB[1, j]
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj, jy + 1)
            i_hi = min(n, j + k)
            iy = ky0 + j * incy
            ix = kx0 + j * incx
            for i in j+1:i_hi
                aij = unsafe_load(A, colj + (i - j) + 1)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += aij * unsafe_load(x, ix + 1)
                iy += incy; ix += incx
            end
            unsafe_store!(y, unsafe_load(y, jy + 1) + α * tmp2, jy + 1)
        end
    end
    return nothing
end

# ─── hbmv: y := α·A·x + β·y    (A Hermitian band, complex) ───────────────

@inline function _hbmv_impl!(uplo::UInt8, n::Int, k::Int, α,
                              A::Ptr{Complex{TR}}, lda::Int,
                              x::Ptr{Complex{TR}}, incx::Int,
                              β, y::Ptr{Complex{TR}}, incy::Int) where {TR<:Real}
    T = Complex{TR}
    n <= 0 && return nothing
    _scale_y!(y, n, incy, β)
    α == zero(α) && return nothing
    kx0 = _start_off(n, incx)
    ky0 = _start_off(n, incy)
    if _isU(uplo)
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            colj = (j - 1) * lda
            i_lo = max(1, j - k)
            iy = ky0 + (i_lo - 1) * incy
            ix = kx0 + (i_lo - 1) * incx
            for i in i_lo:j-1
                aij = unsafe_load(A, colj + (k + i - j) + 1)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += conj(aij) * unsafe_load(x, ix + 1)
                iy += incy; ix += incx
            end
            ajj_re = real(unsafe_load(A, colj + k + 1))
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj_re + α * tmp2, jy + 1)
        end
    else
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            colj = (j - 1) * lda
            ajj_re = real(unsafe_load(A, colj + 1))
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj_re, jy + 1)
            i_hi = min(n, j + k)
            iy = ky0 + j * incy
            ix = kx0 + j * incx
            for i in j+1:i_hi
                aij = unsafe_load(A, colj + (i - j) + 1)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += conj(aij) * unsafe_load(x, ix + 1)
                iy += incy; ix += incx
            end
            unsafe_store!(y, unsafe_load(y, jy + 1) + α * tmp2, jy + 1)
        end
    end
    return nothing
end

# ─── tbmv: x := op(A)·x    (A triangular band, in place) ─────────────────
#
# Same column-walking order as trmv to keep the in-place update safe.
# For uplo='U': A[i,j] at AB[k+1+i-j, j], so the off-diagonal entries on
# column j live at offsets colj+k+i-j+1 for i=max(1,j-k)..j-1, and the
# diagonal at colj+k+1.
# For uplo='L': A[i,j] at AB[1+i-j, j]; off-diagonals at colj+i-j+1 for
# i=j+1..min(n,j+k), diagonal at colj+1.

@inline function _tbmv_impl!(uplo::UInt8, trans::UInt8, diag::UInt8,
                              n::Int, k::Int,
                              A::Ptr{T}, lda::Int,
                              x::Ptr{T}, incx::Int) where {T}
    n <= 0 && return nothing
    nounit = !_isU(diag)
    kx0 = _start_off(n, incx)
    if _isN(trans)
        if _isU(uplo)
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    colj = (j - 1) * lda
                    i_lo = max(1, j - k)
                    ix = kx0 + (i_lo - 1) * incx
                    for i in i_lo:j-1
                        aij = unsafe_load(A, colj + (k + i - j) + 1)
                        unsafe_store!(x, unsafe_load(x, ix + 1) + xj * aij, ix + 1)
                        ix += incx
                    end
                    if nounit
                        unsafe_store!(x, xj * unsafe_load(A, colj + k + 1), jx + 1)
                    end
                end
            end
        else
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    colj = (j - 1) * lda
                    i_hi = min(n, j + k)
                    ix = kx0 + j * incx
                    for i in j+1:i_hi
                        aij = unsafe_load(A, colj + (i - j) + 1)
                        unsafe_store!(x, unsafe_load(x, ix + 1) + xj * aij, ix + 1)
                        ix += incx
                    end
                    if nounit
                        unsafe_store!(x, xj * unsafe_load(A, colj + 1), jx + 1)
                    end
                end
            end
        end
    elseif _isT(trans)
        if _isU(uplo)
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * unsafe_load(A, colj + k + 1)
                end
                i_lo = max(1, j - k)
                ix = kx0 + (i_lo - 1) * incx
                for i in i_lo:j-1
                    tmp += unsafe_load(A, colj + (k + i - j) + 1) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        else
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * unsafe_load(A, colj + 1)
                end
                i_hi = min(n, j + k)
                ix = kx0 + j * incx
                for i in j+1:i_hi
                    tmp += unsafe_load(A, colj + (i - j) + 1) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        end
    else  # 'C'
        if _isU(uplo)
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * conj(unsafe_load(A, colj + k + 1))
                end
                i_lo = max(1, j - k)
                ix = kx0 + (i_lo - 1) * incx
                for i in i_lo:j-1
                    tmp += conj(unsafe_load(A, colj + (k + i - j) + 1)) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        else
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * conj(unsafe_load(A, colj + 1))
                end
                i_hi = min(n, j + k)
                ix = kx0 + j * incx
                for i in j+1:i_hi
                    tmp += conj(unsafe_load(A, colj + (i - j) + 1)) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        end
    end
    return nothing
end

# ─── tbsv: solve op(A)·x = b for x, in place    (A triangular band) ──────

@inline function _tbsv_impl!(uplo::UInt8, trans::UInt8, diag::UInt8,
                              n::Int, k::Int,
                              A::Ptr{T}, lda::Int,
                              x::Ptr{T}, incx::Int) where {T}
    n <= 0 && return nothing
    nounit = !_isU(diag)
    kx0 = _start_off(n, incx)
    if _isN(trans)
        if _isU(uplo)
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    colj = (j - 1) * lda
                    if nounit
                        xj = xj / unsafe_load(A, colj + k + 1)
                        unsafe_store!(x, xj, jx + 1)
                    end
                    i_lo = max(1, j - k)
                    ix = kx0 + (i_lo - 1) * incx
                    for i in i_lo:j-1
                        aij = unsafe_load(A, colj + (k + i - j) + 1)
                        unsafe_store!(x, unsafe_load(x, ix + 1) - xj * aij, ix + 1)
                        ix += incx
                    end
                end
            end
        else
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    colj = (j - 1) * lda
                    if nounit
                        xj = xj / unsafe_load(A, colj + 1)
                        unsafe_store!(x, xj, jx + 1)
                    end
                    i_hi = min(n, j + k)
                    ix = kx0 + j * incx
                    for i in j+1:i_hi
                        aij = unsafe_load(A, colj + (i - j) + 1)
                        unsafe_store!(x, unsafe_load(x, ix + 1) - xj * aij, ix + 1)
                        ix += incx
                    end
                end
            end
        end
    elseif _isT(trans)
        if _isU(uplo)
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                i_lo = max(1, j - k)
                ix = kx0 + (i_lo - 1) * incx
                for i in i_lo:j-1
                    tmp -= unsafe_load(A, colj + (k + i - j) + 1) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                if nounit
                    tmp = tmp / unsafe_load(A, colj + k + 1)
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        else
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                i_hi = min(n, j + k)
                ix = kx0 + j * incx
                for i in j+1:i_hi
                    tmp -= unsafe_load(A, colj + (i - j) + 1) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                if nounit
                    tmp = tmp / unsafe_load(A, colj + 1)
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        end
    else  # 'C'
        if _isU(uplo)
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                i_lo = max(1, j - k)
                ix = kx0 + (i_lo - 1) * incx
                for i in i_lo:j-1
                    tmp -= conj(unsafe_load(A, colj + (k + i - j) + 1)) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                if nounit
                    tmp = tmp / conj(unsafe_load(A, colj + k + 1))
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        else
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                colj = (j - 1) * lda
                tmp = unsafe_load(x, jx + 1)
                i_hi = min(n, j + k)
                ix = kx0 + j * incx
                for i in j+1:i_hi
                    tmp -= conj(unsafe_load(A, colj + (i - j) + 1)) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                if nounit
                    tmp = tmp / conj(unsafe_load(A, colj + 1))
                end
                unsafe_store!(x, tmp, jx + 1)
            end
        end
    end
    return nothing
end

# ─── Packed storage helpers ──────────────────────────────────────────────
#
# Upper packed: A[i,j] (i ≤ j) at AP[(j-1)j/2 + i].
# Lower packed: A[i,j] (i ≥ j) at AP[(j-1)(2n-j+2)/2 + (i-j+1)].
# The kernels track `kk` (1-indexed start of column j's chunk in AP)
# incrementally: kk += j for upper, kk += n-j+1 for lower.

# ─── spmv: y := α·A·x + β·y    (A symmetric, packed real) ────────────────

@inline function _spmv_impl!(uplo::UInt8, n::Int, α,
                              AP::Ptr{T},
                              x::Ptr{T}, incx::Int,
                              β, y::Ptr{T}, incy::Int) where {T<:Real}
    n <= 0 && return nothing
    _scale_y!(y, n, incy, β)
    α == zero(α) && return nothing
    kx0 = _start_off(n, incx)
    ky0 = _start_off(n, incy)
    if _isU(uplo)
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            ix = kx0; iy = ky0
            for i in 1:j-1
                aij = unsafe_load(AP, kk + i - 1)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += aij * unsafe_load(x, ix + 1)
                ix += incx; iy += incy
            end
            ajj = unsafe_load(AP, kk + j - 1)
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj + α * tmp2, jy + 1)
            kk += j
        end
    else
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            ajj = unsafe_load(AP, kk)
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj, jy + 1)
            ix = kx0 + j * incx; iy = ky0 + j * incy
            for i in j+1:n
                aij = unsafe_load(AP, kk + i - j)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += aij * unsafe_load(x, ix + 1)
                ix += incx; iy += incy
            end
            unsafe_store!(y, unsafe_load(y, jy + 1) + α * tmp2, jy + 1)
            kk += n - j + 1
        end
    end
    return nothing
end

# ─── hpmv: y := α·A·x + β·y    (A Hermitian, packed complex) ─────────────

@inline function _hpmv_impl!(uplo::UInt8, n::Int, α,
                              AP::Ptr{Complex{TR}},
                              x::Ptr{Complex{TR}}, incx::Int,
                              β, y::Ptr{Complex{TR}}, incy::Int) where {TR<:Real}
    T = Complex{TR}
    n <= 0 && return nothing
    _scale_y!(y, n, incy, β)
    α == zero(α) && return nothing
    kx0 = _start_off(n, incx)
    ky0 = _start_off(n, incy)
    if _isU(uplo)
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            ix = kx0; iy = ky0
            for i in 1:j-1
                aij = unsafe_load(AP, kk + i - 1)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += conj(aij) * unsafe_load(x, ix + 1)
                ix += incx; iy += incy
            end
            ajj_re = real(unsafe_load(AP, kk + j - 1))
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj_re + α * tmp2, jy + 1)
            kk += j
        end
    else
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            tmp1 = α * unsafe_load(x, jx + 1)
            tmp2 = zero(T)
            ajj_re = real(unsafe_load(AP, kk))
            unsafe_store!(y, unsafe_load(y, jy + 1) + tmp1 * ajj_re, jy + 1)
            ix = kx0 + j * incx; iy = ky0 + j * incy
            for i in j+1:n
                aij = unsafe_load(AP, kk + i - j)
                unsafe_store!(y, unsafe_load(y, iy + 1) + tmp1 * aij, iy + 1)
                tmp2 += conj(aij) * unsafe_load(x, ix + 1)
                ix += incx; iy += incy
            end
            unsafe_store!(y, unsafe_load(y, jy + 1) + α * tmp2, jy + 1)
            kk += n - j + 1
        end
    end
    return nothing
end

# ─── spr: A := α·x·xᵀ + A    (A symmetric, packed real) ──────────────────

@inline function _spr_impl!(uplo::UInt8, n::Int, α,
                             x::Ptr{T}, incx::Int, AP::Ptr{T}) where {T<:Real}
    (n <= 0 || α == zero(α)) && return nothing
    kx0 = _start_off(n, incx)
    if _isU(uplo)
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            xj = unsafe_load(x, jx + 1)
            if xj != zero(T)
                tmp = α * xj
                ix = kx0
                for i in 1:j
                    off = kk + i - 1
                    unsafe_store!(AP, unsafe_load(AP, off) + tmp * unsafe_load(x, ix + 1), off)
                    ix += incx
                end
            end
            kk += j
        end
    else
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            xj = unsafe_load(x, jx + 1)
            if xj != zero(T)
                tmp = α * xj
                ix = jx
                for i in j:n
                    off = kk + i - j
                    unsafe_store!(AP, unsafe_load(AP, off) + tmp * unsafe_load(x, ix + 1), off)
                    ix += incx
                end
            end
            kk += n - j + 1
        end
    end
    return nothing
end

# ─── spr2: A := α·x·yᵀ + α·y·xᵀ + A    (real symmetric, packed) ─────────

@inline function _spr2_impl!(uplo::UInt8, n::Int, α,
                              x::Ptr{T}, incx::Int,
                              y::Ptr{T}, incy::Int,
                              AP::Ptr{T}) where {T<:Real}
    (n <= 0 || α == zero(α)) && return nothing
    kx0 = _start_off(n, incx)
    ky0 = _start_off(n, incy)
    if _isU(uplo)
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            xj = unsafe_load(x, jx + 1)
            yj = unsafe_load(y, jy + 1)
            if xj != zero(T) || yj != zero(T)
                tmp1 = α * yj
                tmp2 = α * xj
                ix = kx0; iy = ky0
                for i in 1:j
                    off = kk + i - 1
                    unsafe_store!(AP,
                        unsafe_load(AP, off) +
                            unsafe_load(x, ix + 1) * tmp1 +
                            unsafe_load(y, iy + 1) * tmp2, off)
                    ix += incx; iy += incy
                end
            end
            kk += j
        end
    else
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            xj = unsafe_load(x, jx + 1)
            yj = unsafe_load(y, jy + 1)
            if xj != zero(T) || yj != zero(T)
                tmp1 = α * yj
                tmp2 = α * xj
                ix = jx; iy = jy
                for i in j:n
                    off = kk + i - j
                    unsafe_store!(AP,
                        unsafe_load(AP, off) +
                            unsafe_load(x, ix + 1) * tmp1 +
                            unsafe_load(y, iy + 1) * tmp2, off)
                    ix += incx; iy += incy
                end
            end
            kk += n - j + 1
        end
    end
    return nothing
end

# ─── hpr: A := α·x·xᴴ + A    (A Hermitian, packed, α real) ───────────────

@inline function _hpr_impl!(uplo::UInt8, n::Int, α::TR,
                             x::Ptr{Complex{TR}}, incx::Int,
                             AP::Ptr{Complex{TR}}) where {TR<:Real}
    T = Complex{TR}
    (n <= 0 || α == zero(α)) && return nothing
    kx0 = _start_off(n, incx)
    if _isU(uplo)
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            xj = unsafe_load(x, jx + 1)
            if xj != zero(T)
                tmp = α * conj(xj)
                ix = kx0
                for i in 1:j-1
                    off = kk + i - 1
                    unsafe_store!(AP, unsafe_load(AP, off) + unsafe_load(x, ix + 1) * tmp, off)
                    ix += incx
                end
                diag_off = kk + j - 1
                ajj = unsafe_load(AP, diag_off)
                unsafe_store!(AP, complex(real(ajj) + real(xj * tmp), zero(TR)), diag_off)
            end
            kk += j
        end
    else
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            xj = unsafe_load(x, jx + 1)
            if xj != zero(T)
                tmp = α * conj(xj)
                ajj = unsafe_load(AP, kk)
                unsafe_store!(AP, complex(real(ajj) + real(xj * tmp), zero(TR)), kk)
                ix = kx0 + j * incx
                for i in j+1:n
                    off = kk + i - j
                    unsafe_store!(AP, unsafe_load(AP, off) + unsafe_load(x, ix + 1) * tmp, off)
                    ix += incx
                end
            end
            kk += n - j + 1
        end
    end
    return nothing
end

# ─── hpr2: A := α·x·yᴴ + ᾱ·y·xᴴ + A    (Hermitian packed, α complex) ────

@inline function _hpr2_impl!(uplo::UInt8, n::Int, α,
                              x::Ptr{Complex{TR}}, incx::Int,
                              y::Ptr{Complex{TR}}, incy::Int,
                              AP::Ptr{Complex{TR}}) where {TR<:Real}
    T = Complex{TR}
    (n <= 0 || α == zero(α)) && return nothing
    kx0 = _start_off(n, incx)
    ky0 = _start_off(n, incy)
    if _isU(uplo)
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            xj = unsafe_load(x, jx + 1)
            yj = unsafe_load(y, jy + 1)
            if xj != zero(T) || yj != zero(T)
                tmp1 = α * conj(yj)
                tmp2 = conj(α) * conj(xj)
                ix = kx0; iy = ky0
                for i in 1:j-1
                    off = kk + i - 1
                    unsafe_store!(AP,
                        unsafe_load(AP, off) +
                            unsafe_load(x, ix + 1) * tmp1 +
                            unsafe_load(y, iy + 1) * tmp2, off)
                    ix += incx; iy += incy
                end
                diag_off = kk + j - 1
                ajj = unsafe_load(AP, diag_off)
                unsafe_store!(AP,
                    complex(real(ajj) + real(xj * tmp1 + yj * tmp2), zero(TR)),
                    diag_off)
            end
            kk += j
        end
    else
        kk = 1
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            xj = unsafe_load(x, jx + 1)
            yj = unsafe_load(y, jy + 1)
            if xj != zero(T) || yj != zero(T)
                tmp1 = α * conj(yj)
                tmp2 = conj(α) * conj(xj)
                ajj = unsafe_load(AP, kk)
                unsafe_store!(AP,
                    complex(real(ajj) + real(xj * tmp1 + yj * tmp2), zero(TR)), kk)
                ix = kx0 + j * incx; iy = ky0 + j * incy
                for i in j+1:n
                    off = kk + i - j
                    unsafe_store!(AP,
                        unsafe_load(AP, off) +
                            unsafe_load(x, ix + 1) * tmp1 +
                            unsafe_load(y, iy + 1) * tmp2, off)
                    ix += incx; iy += incy
                end
            end
            kk += n - j + 1
        end
    end
    return nothing
end

# ─── tpmv: x := op(A)·x    (A triangular, packed, in place) ──────────────

@inline function _tpmv_impl!(uplo::UInt8, trans::UInt8, diag::UInt8,
                              n::Int, AP::Ptr{T},
                              x::Ptr{T}, incx::Int) where {T}
    n <= 0 && return nothing
    nounit = !_isU(diag)
    kx0 = _start_off(n, incx)
    if _isN(trans)
        if _isU(uplo)
            kk = 1
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    ix = kx0
                    for i in 1:j-1
                        unsafe_store!(x,
                            unsafe_load(x, ix + 1) + xj * unsafe_load(AP, kk + i - 1),
                            ix + 1)
                        ix += incx
                    end
                    if nounit
                        unsafe_store!(x, xj * unsafe_load(AP, kk + j - 1), jx + 1)
                    end
                end
                kk += j
            end
        else
            kk = n * (n + 1) ÷ 2
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    ix = kx0 + (n - 1) * incx
                    for i in n:-1:j+1
                        unsafe_store!(x,
                            unsafe_load(x, ix + 1) + xj * unsafe_load(AP, kk - (n - i)),
                            ix + 1)
                        ix -= incx
                    end
                    if nounit
                        unsafe_store!(x, xj * unsafe_load(AP, kk - (n - j)), jx + 1)
                    end
                end
                kk -= n - j + 1
            end
        end
    elseif _isT(trans)
        if _isU(uplo)
            kk = n * (n + 1) ÷ 2
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * unsafe_load(AP, kk)
                end
                ix = kx0 + (j - 2) * incx
                for i in j-1:-1:1
                    tmp += unsafe_load(AP, kk - (j - i)) * unsafe_load(x, ix + 1)
                    ix -= incx
                end
                unsafe_store!(x, tmp, jx + 1)
                kk -= j
            end
        else
            kk = 1
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * unsafe_load(AP, kk)
                end
                ix = kx0 + j * incx
                for i in j+1:n
                    tmp += unsafe_load(AP, kk + i - j) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                unsafe_store!(x, tmp, jx + 1)
                kk += n - j + 1
            end
        end
    else  # 'C'
        if _isU(uplo)
            kk = n * (n + 1) ÷ 2
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * conj(unsafe_load(AP, kk))
                end
                ix = kx0 + (j - 2) * incx
                for i in j-1:-1:1
                    tmp += conj(unsafe_load(AP, kk - (j - i))) * unsafe_load(x, ix + 1)
                    ix -= incx
                end
                unsafe_store!(x, tmp, jx + 1)
                kk -= j
            end
        else
            kk = 1
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                tmp = unsafe_load(x, jx + 1)
                if nounit
                    tmp = tmp * conj(unsafe_load(AP, kk))
                end
                ix = kx0 + j * incx
                for i in j+1:n
                    tmp += conj(unsafe_load(AP, kk + i - j)) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                unsafe_store!(x, tmp, jx + 1)
                kk += n - j + 1
            end
        end
    end
    return nothing
end

# ─── tpsv: solve op(A)·x = b for x, in place    (A triangular packed) ────

@inline function _tpsv_impl!(uplo::UInt8, trans::UInt8, diag::UInt8,
                              n::Int, AP::Ptr{T},
                              x::Ptr{T}, incx::Int) where {T}
    n <= 0 && return nothing
    nounit = !_isU(diag)
    kx0 = _start_off(n, incx)
    if _isN(trans)
        if _isU(uplo)
            kk = n * (n + 1) ÷ 2
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    if nounit
                        xj = xj / unsafe_load(AP, kk)
                        unsafe_store!(x, xj, jx + 1)
                    end
                    ix = kx0 + (j - 2) * incx
                    for i in j-1:-1:1
                        unsafe_store!(x,
                            unsafe_load(x, ix + 1) - xj * unsafe_load(AP, kk - (j - i)),
                            ix + 1)
                        ix -= incx
                    end
                end
                kk -= j
            end
        else
            kk = 1
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                xj = unsafe_load(x, jx + 1)
                if xj != zero(T)
                    if nounit
                        xj = xj / unsafe_load(AP, kk)
                        unsafe_store!(x, xj, jx + 1)
                    end
                    ix = kx0 + j * incx
                    for i in j+1:n
                        unsafe_store!(x,
                            unsafe_load(x, ix + 1) - xj * unsafe_load(AP, kk + i - j),
                            ix + 1)
                        ix += incx
                    end
                end
                kk += n - j + 1
            end
        end
    elseif _isT(trans)
        if _isU(uplo)
            kk = 1
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                tmp = unsafe_load(x, jx + 1)
                ix = kx0
                for i in 1:j-1
                    tmp -= unsafe_load(AP, kk + i - 1) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                if nounit
                    tmp = tmp / unsafe_load(AP, kk + j - 1)
                end
                unsafe_store!(x, tmp, jx + 1)
                kk += j
            end
        else
            kk = n * (n + 1) ÷ 2
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                tmp = unsafe_load(x, jx + 1)
                ix = kx0 + (n - 1) * incx
                for i in n:-1:j+1
                    tmp -= unsafe_load(AP, kk - (n - i)) * unsafe_load(x, ix + 1)
                    ix -= incx
                end
                if nounit
                    tmp = tmp / unsafe_load(AP, kk - (n - j))
                end
                unsafe_store!(x, tmp, jx + 1)
                kk -= n - j + 1
            end
        end
    else  # 'C'
        if _isU(uplo)
            kk = 1
            @inbounds for j in 1:n
                jx = kx0 + (j - 1) * incx
                tmp = unsafe_load(x, jx + 1)
                ix = kx0
                for i in 1:j-1
                    tmp -= conj(unsafe_load(AP, kk + i - 1)) * unsafe_load(x, ix + 1)
                    ix += incx
                end
                if nounit
                    tmp = tmp / conj(unsafe_load(AP, kk + j - 1))
                end
                unsafe_store!(x, tmp, jx + 1)
                kk += j
            end
        else
            kk = n * (n + 1) ÷ 2
            @inbounds for j in n:-1:1
                jx = kx0 + (j - 1) * incx
                tmp = unsafe_load(x, jx + 1)
                ix = kx0 + (n - 1) * incx
                for i in n:-1:j+1
                    tmp -= conj(unsafe_load(AP, kk - (n - i))) * unsafe_load(x, ix + 1)
                    ix -= incx
                end
                if nounit
                    tmp = tmp / conj(unsafe_load(AP, kk - (n - j)))
                end
                unsafe_store!(x, tmp, jx + 1)
                kk -= n - j + 1
            end
        end
    end
    return nothing
end

@inline function _her2_impl!(uplo::UInt8, n::Int, α,
                              x::Ptr{Complex{TR}}, incx::Int,
                              y::Ptr{Complex{TR}}, incy::Int,
                              A::Ptr{Complex{TR}}, lda::Int) where {TR<:Real}
    T = Complex{TR}
    (n <= 0 || α == zero(α)) && return nothing
    kx0 = _start_off(n, incx)
    ky0 = _start_off(n, incy)
    if _isU(uplo)
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            xj = unsafe_load(x, jx + 1)
            yj = unsafe_load(y, jy + 1)
            (xj == zero(T) && yj == zero(T)) && continue
            tmp1 = α * conj(yj)
            tmp2 = conj(α) * conj(xj)
            colj = (j - 1) * lda
            ix = kx0; iy = ky0
            for i in 1:j-1
                off = i + colj
                xi = unsafe_load(x, ix + 1)
                yi = unsafe_load(y, iy + 1)
                unsafe_store!(A, unsafe_load(A, off) + xi * tmp1 + yi * tmp2, off)
                ix += incx; iy += incy
            end
            ajj = unsafe_load(A, j + colj)
            unsafe_store!(A,
                complex(real(ajj) + real(xj * tmp1 + yj * tmp2), zero(TR)),
                j + colj)
        end
    else
        @inbounds for j in 1:n
            jx = kx0 + (j - 1) * incx
            jy = ky0 + (j - 1) * incy
            xj = unsafe_load(x, jx + 1)
            yj = unsafe_load(y, jy + 1)
            (xj == zero(T) && yj == zero(T)) && continue
            tmp1 = α * conj(yj)
            tmp2 = conj(α) * conj(xj)
            colj = (j - 1) * lda
            ajj = unsafe_load(A, j + colj)
            unsafe_store!(A,
                complex(real(ajj) + real(xj * tmp1 + yj * tmp2), zero(TR)),
                j + colj)
            ix = kx0 + j * incx
            iy = ky0 + j * incy
            for i in j+1:n
                off = i + colj
                xi = unsafe_load(x, ix + 1)
                yi = unsafe_load(y, iy + 1)
                unsafe_store!(A, unsafe_load(A, off) + xi * tmp1 + yi * tmp2, off)
                ix += incx; iy += incy
            end
        end
    end
    return nothing
end
