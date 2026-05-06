# End-to-end check that the juliac-compiled libjublas exports Fortran-ABI
# Level-3 BLAS symbols and that {s,d,c,z}gemm_ produce results matching
# Julia's native LinearAlgebra.mul!.
#
# Run after building the library:
#     juliac --output-lib JUBLAS --project=. --trim=safe --experimental \
#            --privatize --bundle build src/JuBLAS.jl
#     julia --project=. test/test_libjublas.jl                     # uses default search list
#     julia --project=. test/test_libjublas.jl /path/to/JUBLAS.so
#
# `--privatize` is required: without it the AOT image's runtime collides
# with the host Julia's runtime in jl_init_threadtls and aborts. juliac
# only accepts `--privatize` together with `--bundle`, so the bundled
# layout (build/lib/JUBLAS.<dlext>) is the canonical output location.
#
# The library is invoked through dlopen + ccall, so this stresses the
# actual ABI surface (LP64 Cint sizes, by-reference scalars, gfortran
# trailing Csize_t length args) — not the in-Julia gemm! method.

using Test
using Libdl
using LinearAlgebra
using Random

const REPO = normpath(joinpath(@__DIR__, ".."))

# Plausible juliac output locations. Override with the JUBLAS_LIB env var
# or by passing an explicit path as the first script argument. The bundled
# layout is listed first since `--privatize` (required to load from Julia)
# only works with `--bundle`.
const SEARCH_PATHS = [
    joinpath(REPO, "build",  "lib", "JUBLAS." * Libdl.dlext), # `juliac --privatize --bundle build …`
    joinpath(REPO, "build",  "JUBLAS." * Libdl.dlext),
    joinpath(REPO, "JUBLAS." * Libdl.dlext),
    joinpath(REPO, "libjublas." * Libdl.dlext),
    joinpath(REPO, "juliac", "libjublas." * Libdl.dlext),
]

function locate_library(explicit::Union{String,Nothing} = nothing)
    explicit !== nothing && return explicit
    env = get(ENV, "JUBLAS_LIB", "")
    isempty(env) || return env
    for p in SEARCH_PATHS
        isfile(p) && return p
    end
    return nothing
end

function open_library(path)
    isfile(path) || error("libjublas not found at $path")
    return Libdl.dlopen(path; throw_error = true)
end

# ────────────────────────────────────────────────────────────────────────
# Fortran-ABI shims. Each ccall mirrors the @ccallable signature in
# src/blas_interface.jl: every scalar by reference, characters as
# Ptr{UInt8} with a trailing Csize_t length per gfortran convention.
# ────────────────────────────────────────────────────────────────────────

const BlasInt = Cint

for (jname, sym, T) in (
        (:sgemm_jublas!, :sgemm_, Float32),
        (:dgemm_jublas!, :dgemm_, Float64),
        (:cgemm_jublas!, :cgemm_, ComplexF32),
        (:zgemm_jublas!, :zgemm_, ComplexF64),
    )
    @eval function $jname(handle, transA::Char, transB::Char,
                          α::$T, A::AbstractMatrix{$T}, B::AbstractMatrix{$T},
                          β::$T, C::AbstractMatrix{$T})
        M, N = size(C)
        K    = size(A, transA == 'N' ? 2 : 1)
        lda  = stride(A, 2)
        ldb  = stride(B, 2)
        ldc  = stride(C, 2)
        ta   = UInt8(transA)
        tb   = UInt8(transB)
        ptr  = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{UInt8},
               Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Csize_t, Csize_t),
              ta, tb,
              BlasInt(M), BlasInt(N), BlasInt(K),
              α, A, BlasInt(lda),
              B, BlasInt(ldb),
              β, C, BlasInt(ldc),
              Csize_t(1), Csize_t(1))
        return C
    end
end

# Reference: α·op(A)·op(B) + β·C0 via Julia's native mul!.
function ref_gemm(transA::Char, transB::Char, α::T, A, B, β::T, C0) where {T}
    Aop = transA == 'N' ? A : transA == 'T' ? transpose(A) : adjoint(A)
    Bop = transB == 'N' ? B : transB == 'T' ? transpose(B) : adjoint(B)
    C   = copy(C0)
    mul!(C, Aop, Bop, α, β)
    return C
end

# Build (A, B) sized so that op(A)*op(B) is M×N with inner dim K.
function make_inputs(::Type{T}, transA::Char, transB::Char, M, N, K, rng) where {T}
    A = transA == 'N' ? randmat(T, M, K, rng) : randmat(T, K, M, rng)
    B = transB == 'N' ? randmat(T, K, N, rng) : randmat(T, N, K, rng)
    return A, B
end

randmat(::Type{T}, m, n, rng) where {T<:Real}    = randn(rng, T, m, n)
randmat(::Type{T}, m, n, rng) where {T<:Complex} = randn(rng, T, m, n)

# ────────────────────────────────────────────────────────────────────────
# BLAS Level 1 — strided helpers + ccall shims.
# ────────────────────────────────────────────────────────────────────────

# Lay out logical vector `v` into a strided buffer per BLAS convention:
#   inc > 0: v[i] at offset (i-1)*|inc|
#   inc < 0: v[i] at offset (n-i)*|inc|   (algorithm walks backward)
function strided_buffer(v::AbstractVector{T}, inc::Int) where {T}
    n = length(v)
    n == 0 && return T[]
    abi = abs(inc)
    buf = zeros(T, (n - 1) * abi + 1)
    for i in 1:n
        offset = inc > 0 ? (i - 1) * abi : (n - i) * abi
        buf[offset + 1] = v[i]
    end
    return buf
end

function extract_strided(buf::AbstractVector{T}, n::Int, inc::Int) where {T}
    n == 0 && return T[]
    abi = abs(inc)
    out = Vector{T}(undef, n)
    for i in 1:n
        offset = inc > 0 ? (i - 1) * abi : (n - i) * abi
        out[i] = buf[offset + 1]
    end
    return out
end

# Generate two-vector ccall shims for axpy / copy / swap (same signature
# pattern: n, x, incx, y, incy — axpy also has α).
for (jname, sym, T) in (
        (:saxpy, :saxpy_, Float32), (:daxpy, :daxpy_, Float64),
        (:caxpy, :caxpy_, ComplexF32), (:zaxpy, :zaxpy_, ComplexF64),
    )
    @eval function $jname(handle, n::Int, α::$T,
                          x::AbstractVector{$T}, incx::Int,
                          y::AbstractVector{$T}, incy::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{BlasInt}, Ref{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}),
              BlasInt(n), α, x, BlasInt(incx), y, BlasInt(incy))
        return y
    end
end

for (jname, sym, T) in (
        (:scopy, :scopy_, Float32), (:dcopy, :dcopy_, Float64),
        (:ccopy, :ccopy_, ComplexF32), (:zcopy, :zcopy_, ComplexF64),
        (:sswap, :sswap_, Float32), (:dswap, :dswap_, Float64),
        (:cswap, :cswap_, ComplexF32), (:zswap, :zswap_, ComplexF64),
    )
    @eval function $jname(handle, n::Int,
                          x::AbstractVector{$T}, incx::Int,
                          y::AbstractVector{$T}, incy::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}),
              BlasInt(n), x, BlasInt(incx), y, BlasInt(incy))
        return nothing
    end
end

# scal: same eltype.
for (jname, sym, T) in (
        (:sscal, :sscal_, Float32), (:dscal, :dscal_, Float64),
        (:cscal, :cscal_, ComplexF32), (:zscal, :zscal_, ComplexF64),
    )
    @eval function $jname(handle, n::Int, α::$T, x::AbstractVector{$T}, incx::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{BlasInt}, Ref{$T}, Ptr{$T}, Ref{BlasInt}),
              BlasInt(n), α, x, BlasInt(incx))
        return x
    end
end

# Mixed-precision scal: real α, complex x.
for (jname, sym, T, TR) in (
        (:csscal, :csscal_, ComplexF32, Float32),
        (:zdscal, :zdscal_, ComplexF64, Float64),
    )
    @eval function $jname(handle, n::Int, α::$TR, x::AbstractVector{$T}, incx::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{BlasInt}, Ref{$TR}, Ptr{$T}, Ref{BlasInt}),
              BlasInt(n), α, x, BlasInt(incx))
        return x
    end
end

# Real dot — returns the scalar.
for (jname, sym, T, TR) in (
        (:sdot,  :sdot_,  Float32, Float32),
        (:ddot,  :ddot_,  Float64, Float64),
        (:dsdot, :dsdot_, Float32, Float64),  # mixed: F32 inputs, F64 result
    )
    @eval function $jname(handle, n::Int,
                          x::AbstractVector{$T}, incx::Int,
                          y::AbstractVector{$T}, incy::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        return ccall(ptr, $TR,
                     (Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}),
                     BlasInt(n), x, BlasInt(incx), y, BlasInt(incy))
    end
end

function sdsdot(handle, n::Int, sb::Float32,
                x::AbstractVector{Float32}, incx::Int,
                y::AbstractVector{Float32}, incy::Int)
    ptr = Libdl.dlsym(handle, :sdsdot_)
    return ccall(ptr, Float32,
                 (Ref{BlasInt}, Ref{Float32}, Ptr{Float32}, Ref{BlasInt},
                  Ptr{Float32}, Ref{BlasInt}),
                 BlasInt(n), sb, x, BlasInt(incx), y, BlasInt(incy))
end

# Complex dot — result via hidden out-pointer first arg.
for (jname, sym_u, sym_c, T) in (
        (:cdot, :cdotu_, :cdotc_, ComplexF32),
        (:zdot, :zdotu_, :zdotc_, ComplexF64),
    )
    sym_u_name = Symbol(jname, "u")
    sym_c_name = Symbol(jname, "c")
    @eval begin
        function $sym_u_name(handle, n::Int,
                             x::AbstractVector{$T}, incx::Int,
                             y::AbstractVector{$T}, incy::Int)
            ptr = Libdl.dlsym(handle, $(QuoteNode(sym_u)))
            r = Ref{$T}()
            ccall(ptr, Cvoid,
                  (Ref{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}),
                  r, BlasInt(n), x, BlasInt(incx), y, BlasInt(incy))
            return r[]
        end

        function $sym_c_name(handle, n::Int,
                             x::AbstractVector{$T}, incx::Int,
                             y::AbstractVector{$T}, incy::Int)
            ptr = Libdl.dlsym(handle, $(QuoteNode(sym_c)))
            r = Ref{$T}()
            ccall(ptr, Cvoid,
                  (Ref{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}),
                  r, BlasInt(n), x, BlasInt(incx), y, BlasInt(incy))
            return r[]
        end
    end
end

# Single-vector reductions: nrm2, asum, iamax.
for (jname, sym, T, TR) in (
        (:snrm2,  :snrm2_,  Float32,    Float32),
        (:dnrm2,  :dnrm2_,  Float64,    Float64),
        (:scnrm2, :scnrm2_, ComplexF32, Float32),
        (:dznrm2, :dznrm2_, ComplexF64, Float64),
        (:sasum,  :sasum_,  Float32,    Float32),
        (:dasum,  :dasum_,  Float64,    Float64),
        (:scasum, :scasum_, ComplexF32, Float32),
        (:dzasum, :dzasum_, ComplexF64, Float64),
    )
    @eval function $jname(handle, n::Int, x::AbstractVector{$T}, incx::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        return ccall(ptr, $TR,
                     (Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}),
                     BlasInt(n), x, BlasInt(incx))
    end
end

for (jname, sym, T) in (
        (:isamax, :isamax_, Float32),
        (:idamax, :idamax_, Float64),
        (:icamax, :icamax_, ComplexF32),
        (:izamax, :izamax_, ComplexF64),
    )
    @eval function $jname(handle, n::Int, x::AbstractVector{$T}, incx::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        return Int(ccall(ptr, BlasInt,
                         (Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}),
                         BlasInt(n), x, BlasInt(incx)))
    end
end

# Givens shims.
for (jname, sym, Tv, Tc) in (
        (:srot,  :srot_,  Float32,    Float32),
        (:drot,  :drot_,  Float64,    Float64),
        (:csrot, :csrot_, ComplexF32, Float32),
        (:zdrot, :zdrot_, ComplexF64, Float64),
    )
    @eval function $jname(handle, n::Int,
                          x::AbstractVector{$Tv}, incx::Int,
                          y::AbstractVector{$Tv}, incy::Int,
                          c::$Tc, s::$Tc)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{BlasInt}, Ptr{$Tv}, Ref{BlasInt},
               Ptr{$Tv}, Ref{BlasInt}, Ref{$Tc}, Ref{$Tc}),
              BlasInt(n), x, BlasInt(incx),
              y, BlasInt(incy), c, s)
        return nothing
    end
end

for (jname, sym, T) in ((:srotg, :srotg_, Float32),
                         (:drotg, :drotg_, Float64))
    @eval function $jname(handle, a::$T, b::$T)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ar = Ref{$T}(a); br = Ref{$T}(b)
        cr = Ref{$T}(); sr = Ref{$T}()
        ccall(ptr, Cvoid,
              (Ref{$T}, Ref{$T}, Ref{$T}, Ref{$T}),
              ar, br, cr, sr)
        return ar[], br[], cr[], sr[]
    end
end

for (jname, sym, T, TR) in ((:crotg, :crotg_, ComplexF32, Float32),
                             (:zrotg, :zrotg_, ComplexF64, Float64))
    @eval function $jname(handle, a::$T, b::$T)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ar = Ref{$T}(a); br = Ref{$T}(b)
        cr = Ref{$TR}(); sr = Ref{$T}()
        ccall(ptr, Cvoid,
              (Ref{$T}, Ref{$T}, Ref{$TR}, Ref{$T}),
              ar, br, cr, sr)
        return ar[], br[], cr[], sr[]
    end
end

for (jname, sym, T) in ((:srotm, :srotm_, Float32),
                         (:drotm, :drotm_, Float64))
    @eval function $jname(handle, n::Int,
                          x::AbstractVector{$T}, incx::Int,
                          y::AbstractVector{$T}, incy::Int,
                          param::AbstractVector{$T})
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt}, Ptr{$T}),
              BlasInt(n), x, BlasInt(incx),
              y, BlasInt(incy), param)
        return nothing
    end
end

for (jname, sym, T) in ((:srotmg, :srotmg_, Float32),
                         (:drotmg, :drotmg_, Float64))
    @eval function $jname(handle, d1::$T, d2::$T, x1::$T, y1::$T)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        d1r = Ref{$T}(d1); d2r = Ref{$T}(d2)
        x1r = Ref{$T}(x1); y1r = Ref{$T}(y1)
        param = zeros($T, 5)
        ccall(ptr, Cvoid,
              (Ref{$T}, Ref{$T}, Ref{$T}, Ref{$T}, Ptr{$T}),
              d1r, d2r, x1r, y1r, param)
        return d1r[], d2r[], x1r[], y1r[], param
    end
end

# ────────────────────────────────────────────────────────────────────────
# Test sweep.
# ────────────────────────────────────────────────────────────────────────

const SHAPES   = [(1, 1, 1), (4, 4, 4), (32, 17, 23), (128, 64, 96)]
const REAL_TR  = ['N', 'T']
const CPLX_TR  = ['N', 'T', 'C']

# (n, incx, incy) sweep for two-vector L1 routines. Covers unit, multi-,
# negative, and mixed-sign strides.
const STRIDES_2 = [
    (1,    1,  1),
    (8,    1,  1),
    (17,   1,  1),
    (256,  1,  1),
    (17,   2,  3),
    (17,  -1,  1),
    (17,   1, -1),
    (17,  -2,  3),
]

# Reference: BLAS conventions for dot/asum/iamax on real and complex inputs.
ref_dot(x, y)  = sum(xi * yi       for (xi, yi) in zip(x, y); init = zero(eltype(x)) * zero(eltype(y)))
ref_dotc(x, y) = sum(conj(xi) * yi for (xi, yi) in zip(x, y); init = zero(eltype(x)) * zero(eltype(y)))
ref_asum(x::AbstractVector{<:Real})    = sum(abs, x; init = zero(eltype(x)))
ref_asum(x::AbstractVector{<:Complex}) = sum(v -> abs(real(v)) + abs(imag(v)), x;
                                              init = real(zero(eltype(x))))
ref_iamax(x::AbstractVector{<:Real})    = isempty(x) ? 0 : argmax(abs.(x))
ref_iamax(x::AbstractVector{<:Complex}) = isempty(x) ? 0 :
    argmax(abs(real(v)) + abs(imag(v)) for v in x)

rand_vec(::Type{T}, n, rng) where {T} = randn(rng, T, n)

make_alpha(::Type{Float32})    = 0.7f0
make_alpha(::Type{Float64})    = 0.7
make_alpha(::Type{ComplexF32}) = 0.7f0 - 0.3f0im
make_alpha(::Type{ComplexF64}) = 0.7   - 0.3im

make_beta(::Type{Float32})    = -0.3f0
make_beta(::Type{Float64})    = -0.3
make_beta(::Type{ComplexF32}) =  0.5f0 + 0.2f0im
make_beta(::Type{ComplexF64}) =  0.5   + 0.2im

# Build an `lda × n` storage matrix containing `A_logical` (m×n) in the
# first m rows; the trailing `lda - m` rows are random padding that the
# routine must leave untouched.
function strided_matrix(A_logical::AbstractMatrix{T}, lda::Int, rng) where {T}
    m, n = size(A_logical)
    @assert lda >= m
    M = randn(rng, T, lda, n)
    M[1:m, :] .= A_logical
    return M
end

extract_matrix(M::AbstractMatrix, m::Int, n::Int) = M[1:m, 1:n]

# ────────────────────────────────────────────────────────────────────────
# BLAS Level 2 — gemv + ger family ccall shims.
# ────────────────────────────────────────────────────────────────────────

for (jname, sym, T) in (
        (:sgemv, :sgemv_, Float32),
        (:dgemv, :dgemv_, Float64),
        (:cgemv, :cgemv_, ComplexF32),
        (:zgemv, :zgemv_, ComplexF64),
    )
    @eval function $jname(handle, trans::Char,
                          m::Int, n::Int, α::$T,
                          A::AbstractMatrix{$T}, lda::Int,
                          x::AbstractVector{$T}, incx::Int,
                          β::$T,
                          y::AbstractVector{$T}, incy::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8},
               Ref{BlasInt}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Csize_t),
              UInt8(trans),
              BlasInt(m), BlasInt(n),
              α, A, BlasInt(lda),
              x, BlasInt(incx),
              β, y, BlasInt(incy),
              Csize_t(1))
        return y
    end
end

for (jname, sym, T) in (
        (:sger,  :sger_,  Float32),
        (:dger,  :dger_,  Float64),
        (:cgeru, :cgeru_, ComplexF32),
        (:zgeru, :zgeru_, ComplexF64),
        (:cgerc, :cgerc_, ComplexF32),
        (:zgerc, :zgerc_, ComplexF64),
    )
    @eval function $jname(handle, m::Int, n::Int, α::$T,
                          x::AbstractVector{$T}, incx::Int,
                          y::AbstractVector{$T}, incy::Int,
                          A::AbstractMatrix{$T}, lda::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{BlasInt}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt}),
              BlasInt(m), BlasInt(n),
              α, x, BlasInt(incx),
              y, BlasInt(incy),
              A, BlasInt(lda))
        return A
    end
end

# symv / hemv shims.
for (jname, sym, T) in (
        (:ssymv, :ssymv_, Float32),    (:dsymv, :dsymv_, Float64),
        (:chemv, :chemv_, ComplexF32), (:zhemv, :zhemv_, ComplexF64),
    )
    @eval function $jname(handle, uplo::Char, n::Int, α::$T,
                          A::AbstractMatrix{$T}, lda::Int,
                          x::AbstractVector{$T}, incx::Int,
                          β::$T, y::AbstractVector{$T}, incy::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt}, Csize_t),
              UInt8(uplo), BlasInt(n),
              α, A, BlasInt(lda),
              x, BlasInt(incx),
              β, y, BlasInt(incy), Csize_t(1))
        return y
    end
end

# syr / her shims (her takes real α).
for (jname, sym, T) in ((:ssyr, :ssyr_, Float32),
                         (:dsyr, :dsyr_, Float64))
    @eval function $jname(handle, uplo::Char, n::Int, α::$T,
                          x::AbstractVector{$T}, incx::Int,
                          A::AbstractMatrix{$T}, lda::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt}, Csize_t),
              UInt8(uplo), BlasInt(n),
              α, x, BlasInt(incx),
              A, BlasInt(lda), Csize_t(1))
        return A
    end
end

for (jname, sym, T, TR) in ((:cher, :cher_, ComplexF32, Float32),
                             (:zher, :zher_, ComplexF64, Float64))
    @eval function $jname(handle, uplo::Char, n::Int, α::$TR,
                          x::AbstractVector{$T}, incx::Int,
                          A::AbstractMatrix{$T}, lda::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{BlasInt},
               Ref{$TR}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt}, Csize_t),
              UInt8(uplo), BlasInt(n),
              α, x, BlasInt(incx),
              A, BlasInt(lda), Csize_t(1))
        return A
    end
end

# syr2 / her2 shims.
for (jname, sym, T) in (
        (:ssyr2, :ssyr2_, Float32),    (:dsyr2, :dsyr2_, Float64),
        (:cher2, :cher2_, ComplexF32), (:zher2, :zher2_, ComplexF64),
    )
    @eval function $jname(handle, uplo::Char, n::Int, α::$T,
                          x::AbstractVector{$T}, incx::Int,
                          y::AbstractVector{$T}, incy::Int,
                          A::AbstractMatrix{$T}, lda::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt}, Csize_t),
              UInt8(uplo), BlasInt(n),
              α, x, BlasInt(incx),
              y, BlasInt(incy),
              A, BlasInt(lda), Csize_t(1))
        return A
    end
end

# trmv / trsv shims (same signature, 3 hidden char-length args).
for (jname, sym, T) in (
        (:strmv, :strmv_, Float32),    (:dtrmv, :dtrmv_, Float64),
        (:ctrmv, :ctrmv_, ComplexF32), (:ztrmv, :ztrmv_, ComplexF64),
        (:strsv, :strsv_, Float32),    (:dtrsv, :dtrsv_, Float64),
        (:ctrsv, :ctrsv_, ComplexF32), (:ztrsv, :ztrsv_, ComplexF64),
    )
    @eval function $jname(handle, uplo::Char, trans::Char, diag::Char,
                          n::Int, A::AbstractMatrix{$T}, lda::Int,
                          x::AbstractVector{$T}, incx::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
               Csize_t, Csize_t, Csize_t),
              UInt8(uplo), UInt8(trans), UInt8(diag), BlasInt(n),
              A, BlasInt(lda), x, BlasInt(incx),
              Csize_t(1), Csize_t(1), Csize_t(1))
        return x
    end
end

# ── Banded shims ──

# gbmv: trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy + 1 hidden len
for (jname, sym, T) in (
        (:sgbmv, :sgbmv_, Float32),    (:dgbmv, :dgbmv_, Float64),
        (:cgbmv, :cgbmv_, ComplexF32), (:zgbmv, :zgbmv_, ComplexF64),
    )
    @eval function $jname(handle, trans::Char, m::Int, n::Int,
                          kl::Int, ku::Int, α::$T,
                          A::AbstractMatrix{$T}, lda::Int,
                          x::AbstractVector{$T}, incx::Int,
                          β::$T, y::AbstractVector{$T}, incy::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
               Ref{BlasInt}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt}, Csize_t),
              UInt8(trans), BlasInt(m), BlasInt(n),
              BlasInt(kl), BlasInt(ku),
              α, A, BlasInt(lda),
              x, BlasInt(incx),
              β, y, BlasInt(incy), Csize_t(1))
        return y
    end
end

# sbmv / hbmv: uplo, n, k, alpha, A, lda, x, incx, beta, y, incy + 1 hidden len
for (jname, sym, T) in (
        (:ssbmv, :ssbmv_, Float32),    (:dsbmv, :dsbmv_, Float64),
        (:chbmv, :chbmv_, ComplexF32), (:zhbmv, :zhbmv_, ComplexF64),
    )
    @eval function $jname(handle, uplo::Char, n::Int, k::Int, α::$T,
                          A::AbstractMatrix{$T}, lda::Int,
                          x::AbstractVector{$T}, incx::Int,
                          β::$T, y::AbstractVector{$T}, incy::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt}, Csize_t),
              UInt8(uplo), BlasInt(n), BlasInt(k),
              α, A, BlasInt(lda),
              x, BlasInt(incx),
              β, y, BlasInt(incy), Csize_t(1))
        return y
    end
end

# tbmv / tbsv: uplo, trans, diag, n, k, A, lda, x, incx + 3 hidden lens
for (jname, sym, T) in (
        (:stbmv, :stbmv_, Float32),    (:dtbmv, :dtbmv_, Float64),
        (:ctbmv, :ctbmv_, ComplexF32), (:ztbmv, :ztbmv_, ComplexF64),
        (:stbsv, :stbsv_, Float32),    (:dtbsv, :dtbsv_, Float64),
        (:ctbsv, :ctbsv_, ComplexF32), (:ztbsv, :ztbsv_, ComplexF64),
    )
    @eval function $jname(handle, uplo::Char, trans::Char, diag::Char,
                          n::Int, k::Int,
                          A::AbstractMatrix{$T}, lda::Int,
                          x::AbstractVector{$T}, incx::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
               Ref{BlasInt}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
               Csize_t, Csize_t, Csize_t),
              UInt8(uplo), UInt8(trans), UInt8(diag),
              BlasInt(n), BlasInt(k),
              A, BlasInt(lda), x, BlasInt(incx),
              Csize_t(1), Csize_t(1), Csize_t(1))
        return x
    end
end

# Band-storage helpers.

# Convert dense `A_dense` (m×n, with non-zeros only in the kl-sub/ku-super
# band) into the Fortran band storage format AB of shape lda×n where
# AB[ku+1+i-j, j] = A[i,j] for max(1,j-ku) ≤ i ≤ min(m,j+kl).
function dense_to_gb(A_dense::AbstractMatrix{T}, kl::Int, ku::Int, lda::Int) where {T}
    m, n = size(A_dense)
    @assert lda >= kl + ku + 1
    AB = zeros(T, lda, n)
    for j in 1:n, i in max(1, j-ku):min(m, j+kl)
        AB[ku + 1 + i - j, j] = A_dense[i, j]
    end
    return AB
end

# Symmetric/Hermitian/triangular upper band: AB[k+1+i-j, j] = A[i,j].
function dense_to_band_U(A_dense::AbstractMatrix{T}, k::Int, lda::Int) where {T}
    n = size(A_dense, 1)
    @assert lda >= k + 1
    AB = zeros(T, lda, n)
    for j in 1:n, i in max(1, j-k):j
        AB[k + 1 + i - j, j] = A_dense[i, j]
    end
    return AB
end

# Lower band: AB[1+i-j, j] = A[i,j].
function dense_to_band_L(A_dense::AbstractMatrix{T}, k::Int, lda::Int) where {T}
    n = size(A_dense, 1)
    @assert lda >= k + 1
    AB = zeros(T, lda, n)
    for j in 1:n, i in j:min(n, j+k)
        AB[1 + i - j, j] = A_dense[i, j]
    end
    return AB
end

# Build a random dense matrix with non-zeros only in the (kl, ku)-band.
function rand_band_dense(::Type{T}, m::Int, n::Int, kl::Int, ku::Int, rng) where {T}
    A = zeros(T, m, n)
    for j in 1:n, i in max(1, j-ku):min(m, j+kl)
        A[i, j] = randn(rng, T)
    end
    return A
end

# Random symmetric / Hermitian band matrix (full dense form, with the
# Symmetric/Hermitian invariant enforced explicitly).
function rand_sym_band_dense(::Type{T}, n::Int, k::Int, rng) where {T<:Real}
    A = zeros(T, n, n)
    for j in 1:n, i in max(1, j-k):j
        v = randn(rng, T)
        A[i, j] = v
        i != j && (A[j, i] = v)
    end
    return A
end

function rand_herm_band_dense(::Type{T}, n::Int, k::Int, rng) where {T<:Complex}
    TR = real(T)
    A = zeros(T, n, n)
    for j in 1:n
        A[j, j] = T(randn(rng, TR))  # diagonal real
        for i in max(1, j-k):j-1
            v = randn(rng, T)
            A[i, j] = v
            A[j, i] = conj(v)
        end
    end
    return A
end

# Diagonally-dominant random triangular band (for tbsv stability).
function rand_tri_band_dense(::Type{T}, n::Int, k::Int, uplo::Char, rng) where {T}
    A = zeros(T, n, n)
    if uplo == 'U'
        for j in 1:n, i in max(1, j-k):j
            A[i, j] = randn(rng, T) / T(n)
            i == j && (A[i, j] += T(1))
        end
    else
        for j in 1:n, i in j:min(n, j+k)
            A[i, j] = randn(rng, T) / T(n)
            i == j && (A[i, j] += T(1))
        end
    end
    return A
end

# ── Packed-storage helpers ──

# Pack the upper triangle of A column-by-column into AP (length n(n+1)/2).
# AP[(j-1)j/2 + i] = A[i,j] for i ≤ j.
function dense_to_packed_U(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    AP = Vector{T}(undef, n * (n + 1) ÷ 2)
    kk = 1
    for j in 1:n
        for i in 1:j
            AP[kk + i - 1] = A[i, j]
        end
        kk += j
    end
    return AP
end

# Pack the lower triangle column-by-column. AP[kk + (i-j)] = A[i,j] for i ≥ j.
function dense_to_packed_L(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    AP = Vector{T}(undef, n * (n + 1) ÷ 2)
    kk = 1
    for j in 1:n
        for i in j:n
            AP[kk + i - j] = A[i, j]
        end
        kk += n - j + 1
    end
    return AP
end

# Inverse of the above — read AP back into a dense (full) matrix mirroring
# the symmetric/Hermitian invariant. `mirror` = `:sym` (copy) or `:herm`
# (conjugate copy) for the unstored triangle.
function packed_to_dense(AP::AbstractVector{T}, n::Int, uplo::Char, mirror::Symbol) where {T}
    A = zeros(T, n, n)
    if uplo == 'U'
        kk = 1
        for j in 1:n
            for i in 1:j
                A[i, j] = AP[kk + i - 1]
                if i != j
                    A[j, i] = mirror === :herm ? conj(A[i, j]) : A[i, j]
                end
            end
            kk += j
        end
    else
        kk = 1
        for j in 1:n
            for i in j:n
                A[i, j] = AP[kk + i - j]
                if i != j
                    A[j, i] = mirror === :herm ? conj(A[i, j]) : A[i, j]
                end
            end
            kk += n - j + 1
        end
    end
    return A
end

# spmv / hpmv shims.
for (jname, sym, T) in (
        (:sspmv, :sspmv_, Float32),    (:dspmv, :dspmv_, Float64),
        (:chpmv, :chpmv_, ComplexF32), (:zhpmv, :zhpmv_, ComplexF64),
    )
    @eval function $jname(handle, uplo::Char, n::Int, α::$T,
                          AP::AbstractVector{$T},
                          x::AbstractVector{$T}, incx::Int,
                          β::$T, y::AbstractVector{$T}, incy::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{BlasInt},
               Ref{$T}, Ptr{$T},
               Ptr{$T}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt}, Csize_t),
              UInt8(uplo), BlasInt(n),
              α, AP,
              x, BlasInt(incx),
              β, y, BlasInt(incy), Csize_t(1))
        return y
    end
end

# spr / hpr shims (hpr takes real α).
for (jname, sym, T) in ((:sspr, :sspr_, Float32),
                         (:dspr, :dspr_, Float64))
    @eval function $jname(handle, uplo::Char, n::Int, α::$T,
                          x::AbstractVector{$T}, incx::Int,
                          AP::AbstractVector{$T})
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Csize_t),
              UInt8(uplo), BlasInt(n),
              α, x, BlasInt(incx),
              AP, Csize_t(1))
        return AP
    end
end

for (jname, sym, T, TR) in ((:chpr, :chpr_, ComplexF32, Float32),
                             (:zhpr, :zhpr_, ComplexF64, Float64))
    @eval function $jname(handle, uplo::Char, n::Int, α::$TR,
                          x::AbstractVector{$T}, incx::Int,
                          AP::AbstractVector{$T})
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{BlasInt},
               Ref{$TR}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Csize_t),
              UInt8(uplo), BlasInt(n),
              α, x, BlasInt(incx),
              AP, Csize_t(1))
        return AP
    end
end

# spr2 / hpr2 shims.
for (jname, sym, T) in (
        (:sspr2, :sspr2_, Float32),    (:dspr2, :dspr2_, Float64),
        (:chpr2, :chpr2_, ComplexF32), (:zhpr2, :zhpr2_, ComplexF64),
    )
    @eval function $jname(handle, uplo::Char, n::Int, α::$T,
                          x::AbstractVector{$T}, incx::Int,
                          y::AbstractVector{$T}, incy::Int,
                          AP::AbstractVector{$T})
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{BlasInt},
               Ref{$T}, Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Ref{BlasInt},
               Ptr{$T}, Csize_t),
              UInt8(uplo), BlasInt(n),
              α, x, BlasInt(incx),
              y, BlasInt(incy),
              AP, Csize_t(1))
        return AP
    end
end

# tpmv / tpsv shims.
for (jname, sym, T) in (
        (:stpmv, :stpmv_, Float32),    (:dtpmv, :dtpmv_, Float64),
        (:ctpmv, :ctpmv_, ComplexF32), (:ztpmv, :ztpmv_, ComplexF64),
        (:stpsv, :stpsv_, Float32),    (:dtpsv, :dtpsv_, Float64),
        (:ctpsv, :ctpsv_, ComplexF32), (:ztpsv, :ztpsv_, ComplexF64),
    )
    @eval function $jname(handle, uplo::Char, trans::Char, diag::Char,
                          n::Int, AP::AbstractVector{$T},
                          x::AbstractVector{$T}, incx::Int)
        ptr = Libdl.dlsym(handle, $(QuoteNode(sym)))
        ccall(ptr, Cvoid,
              (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
               Ptr{$T}, Ptr{$T}, Ref{BlasInt},
               Csize_t, Csize_t, Csize_t),
              UInt8(uplo), UInt8(trans), UInt8(diag), BlasInt(n),
              AP, x, BlasInt(incx),
              Csize_t(1), Csize_t(1), Csize_t(1))
        return x
    end
end

function run_blas1_tests(handle, rng)
    @testset "BLAS Level 1" begin
        # ── axpy (y := α·x + y) ──
        @testset "axpy" begin
            for (jcall, T, atol) in (
                    (saxpy, Float32,    1f-5),
                    (daxpy, Float64,    1e-12),
                    (caxpy, ComplexF32, 1f-5),
                    (zaxpy, ComplexF64, 1e-12),
                )
                @testset "$(nameof(jcall)) $T" begin
                    α = make_alpha(T)
                    for (n, incx, incy) in STRIDES_2
                        x = rand_vec(T, n, rng); y = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx); ybuf = strided_buffer(y, incy)
                        jcall(handle, n, α, xbuf, incx, ybuf, incy)
                        @test isapprox(extract_strided(ybuf, n, incy), y .+ α .* x;
                                       atol = atol * max(1, n))
                    end
                end
            end
        end

        # ── copy (y := x) ──
        @testset "copy" begin
            for (jcall, T) in ((scopy, Float32), (dcopy, Float64),
                                (ccopy, ComplexF32), (zcopy, ComplexF64))
                @testset "$(nameof(jcall)) $T" begin
                    for (n, incx, incy) in STRIDES_2
                        x = rand_vec(T, n, rng); y = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx); ybuf = strided_buffer(y, incy)
                        jcall(handle, n, xbuf, incx, ybuf, incy)
                        @test extract_strided(ybuf, n, incy) == x
                    end
                end
            end
        end

        # ── swap (x ↔ y) ──
        @testset "swap" begin
            for (jcall, T) in ((sswap, Float32), (dswap, Float64),
                                (cswap, ComplexF32), (zswap, ComplexF64))
                @testset "$(nameof(jcall)) $T" begin
                    for (n, incx, incy) in STRIDES_2
                        x = rand_vec(T, n, rng); x0 = copy(x)
                        y = rand_vec(T, n, rng); y0 = copy(y)
                        xbuf = strided_buffer(x, incx); ybuf = strided_buffer(y, incy)
                        jcall(handle, n, xbuf, incx, ybuf, incy)
                        @test extract_strided(xbuf, n, incx) == y0
                        @test extract_strided(ybuf, n, incy) == x0
                    end
                end
            end
        end

        # ── scal (x := α·x) — reference BLAS requires incx > 0. ──
        @testset "scal" begin
            for (jcall, T) in ((sscal, Float32),    (dscal, Float64),
                                (cscal, ComplexF32), (zscal, ComplexF64))
                atol = real(T) === Float32 ? 1f-5 : 1e-12
                @testset "$(nameof(jcall)) $T" begin
                    α = make_alpha(T)
                    for (n, incx) in ((1,1),(8,1),(17,1),(256,1),(17,2),(17,3))
                        x = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        jcall(handle, n, α, xbuf, incx)
                        @test isapprox(extract_strided(xbuf, n, incx), α .* x;
                                       atol = atol * max(1, n))
                    end
                end
            end

            for (jcall, T, αval) in ((csscal, ComplexF32, 0.7f0),
                                      (zdscal, ComplexF64, 0.7))
                atol = real(T) === Float32 ? 1f-5 : 1e-12
                @testset "$(nameof(jcall)) $T" begin
                    for (n, incx) in ((1,1),(8,1),(17,1),(17,2))
                        x = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        jcall(handle, n, αval, xbuf, incx)
                        @test isapprox(extract_strided(xbuf, n, incx), αval .* x;
                                       atol = atol * max(1, n))
                    end
                end
            end
        end

        # ── dot ──
        @testset "dot" begin
            for (jcall, T, atol) in ((sdot, Float32, 1f-4), (ddot, Float64, 1e-12))
                @testset "$(nameof(jcall)) $T" begin
                    for (n, incx, incy) in STRIDES_2
                        x = rand_vec(T, n, rng); y = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx); ybuf = strided_buffer(y, incy)
                        got = jcall(handle, n, xbuf, incx, ybuf, incy)
                        @test isapprox(got, ref_dot(x, y); atol = atol * max(1, n))
                    end
                end
            end

            @testset "dsdot Float32→Float64" begin
                for (n, incx, incy) in STRIDES_2
                    x = rand_vec(Float32, n, rng); y = rand_vec(Float32, n, rng)
                    xbuf = strided_buffer(x, incx); ybuf = strided_buffer(y, incy)
                    got = dsdot(handle, n, xbuf, incx, ybuf, incy)
                    @test isapprox(got, sum(Float64.(x) .* Float64.(y));
                                   atol = 1e-6 * max(1, n))
                end
            end

            @testset "sdsdot bias + Float64-acc" begin
                sb = 0.5f0
                for (n, incx, incy) in STRIDES_2
                    x = rand_vec(Float32, n, rng); y = rand_vec(Float32, n, rng)
                    xbuf = strided_buffer(x, incx); ybuf = strided_buffer(y, incy)
                    got = sdsdot(handle, n, sb, xbuf, incx, ybuf, incy)
                    ref = Float32(Float64(sb) + sum(Float64.(x) .* Float64.(y)))
                    @test isapprox(got, ref; atol = 1f-4 * max(1, n))
                end
            end

            for (jcall_u, jcall_c, T, atol) in (
                    (cdotu, cdotc, ComplexF32, 1f-4),
                    (zdotu, zdotc, ComplexF64, 1e-12),
                )
                @testset "$(nameof(jcall_u))/$(nameof(jcall_c)) $T" begin
                    for (n, incx, incy) in STRIDES_2
                        x = rand_vec(T, n, rng); y = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx); ybuf = strided_buffer(y, incy)
                        got_u = jcall_u(handle, n, xbuf, incx, ybuf, incy)
                        got_c = jcall_c(handle, n, xbuf, incx, ybuf, incy)
                        @test isapprox(got_u, ref_dot(x, y);  atol = atol * max(1, n))
                        @test isapprox(got_c, ref_dotc(x, y); atol = atol * max(1, n))
                    end
                end
            end
        end

        # ── nrm2 / asum / iamax (single-vector) ──
        @testset "nrm2" begin
            for (jcall, T, atol) in (
                    (snrm2,  Float32,    1f-5),
                    (dnrm2,  Float64,    1e-12),
                    (scnrm2, ComplexF32, 1f-5),
                    (dznrm2, ComplexF64, 1e-12),
                )
                @testset "$(nameof(jcall)) $T" begin
                    # Reference BLAS specifies nrm2 returns 0 for incx<1; only
                    # sweep positive strides to match.
                    for (n, incx) in ((1,1),(8,1),(17,1),(256,1),(17,2),(17,3))
                        x = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        got = jcall(handle, n, xbuf, incx)
                        @test isapprox(got, sqrt(sum(abs2, x)); atol = atol * max(1, n))
                    end
                end
            end
        end

        @testset "asum" begin
            for (jcall, T, atol) in (
                    (sasum,  Float32,    1f-5),
                    (dasum,  Float64,    1e-12),
                    (scasum, ComplexF32, 1f-5),
                    (dzasum, ComplexF64, 1e-12),
                )
                @testset "$(nameof(jcall)) $T" begin
                    for (n, incx) in ((1,1),(8,1),(17,1),(256,1),(17,2))
                        x = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        got = jcall(handle, n, xbuf, incx)
                        @test isapprox(got, ref_asum(x); atol = atol * max(1, n))
                    end
                end
            end
        end

        @testset "iamax" begin
            for (jcall, T) in ((isamax, Float32), (idamax, Float64),
                                (icamax, ComplexF32), (izamax, ComplexF64))
                @testset "$(nameof(jcall)) $T" begin
                    for (n, incx) in ((1,1),(8,1),(17,1),(256,1),(17,2),(17,3))
                        x = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        got = jcall(handle, n, xbuf, incx)
                        @test got == ref_iamax(x)
                    end
                end
            end
        end

        # ── Givens family ──
        @testset "rot (apply)" begin
            for (jcall, Tv, Tc, atol) in (
                    (srot,  Float32,    Float32, 1f-5),
                    (drot,  Float64,    Float64, 1e-12),
                    (csrot, ComplexF32, Float32, 1f-5),
                    (zdrot, ComplexF64, Float64, 1e-12),
                )
                @testset "$(nameof(jcall)) $Tv" begin
                    for (n, incx, incy) in STRIDES_2
                        x = rand_vec(Tv, n, rng); y = rand_vec(Tv, n, rng)
                        # Real Givens: c² + s² = 1.
                        θ = Tc(0.7)
                        c = cos(θ); s = sin(θ)
                        xbuf = strided_buffer(x, incx)
                        ybuf = strided_buffer(y, incy)
                        jcall(handle, n, xbuf, incx, ybuf, incy, c, s)
                        @test isapprox(extract_strided(xbuf, n, incx),  c .* x .+ s .* y;
                                       atol = atol * max(1, n))
                        @test isapprox(extract_strided(ybuf, n, incy), -s .* x .+ c .* y;
                                       atol = atol * max(1, n))
                    end
                end
            end
        end

        @testset "rotg (real)" begin
            for (jcall, T, atol) in ((srotg, Float32, 1f-5),
                                       (drotg, Float64, 1e-12))
                @testset "$(nameof(jcall)) $T" begin
                    pairs = [(T(1), T(0)),  (T(0), T(1)),  (T(0), T(0)),
                             (T(3), T(4)),  (T(-3), T(4)), (T(3), T(-4)),
                             (T(1e3), T(1e-3))]
                    for (a0, b0) in pairs
                        a, b, c, s = jcall(handle, a0, b0)
                        # c² + s² ≈ 1 (degenerate when a0 = b0 = 0)
                        if !(a0 == b0 == zero(T))
                            @test isapprox(c*c + s*s, one(T); atol = atol)
                        end
                        # Apply (c, s) to original (a0, b0): first component
                        # should be the returned r (in `a`); second should be 0.
                        @test isapprox( c*a0 + s*b0, a; atol = atol * max(one(T), abs(a)))
                        @test isapprox(-s*a0 + c*b0, zero(T); atol = atol * max(one(T), abs(a)))
                    end
                end
            end
        end

        @testset "rotg (complex)" begin
            for (jcall, T, TR, atol) in ((crotg, ComplexF32, Float32, 1f-5),
                                          (zrotg, ComplexF64, Float64, 1e-12))
                @testset "$(nameof(jcall)) $T" begin
                    pairs = [(T(1+0im),    T(0)),
                             (T(0),        T(2-1im)),
                             (T(3+4im),    T(1-2im)),
                             (T(-1+0.5im), T(0.7+0.3im))]
                    for (a0, b0) in pairs
                        a, _, c, s = jcall(handle, a0, b0)
                        # c² + |s|² ≈ 1   (c is real)
                        @test isapprox(c*c + abs2(s), one(TR); atol = atol)
                        # Rotation must annihilate b0:  conj(s)·a0 - c·b0 ≈ 0,
                        # and  c·a0 + s·b0 ≈ a (the returned r).
                        @test isapprox( c*a0 + s*b0, a; atol = atol * max(one(TR), abs(a)))
                        @test isapprox(-conj(s)*a0 + c*b0, zero(T);
                                       atol = atol * max(one(TR), abs(a)))
                    end
                end
            end
        end

        @testset "rotm + rotmg (modified Givens)" begin
            for (jcall_mg, jcall_m, T, atol) in (
                    (srotmg, srotm, Float32, 1f-4),
                    (drotmg, drotm, Float64, 1e-10),
                )
                @testset "$(nameof(jcall_mg)) / $(nameof(jcall_m)) $T" begin
                    cases = [(T(1.0), T(1.0), T(2.0), T(1.0)),
                             (T(2.0), T(0.5), T(1.0), T(3.0)),
                             (T(1.0), T(1.0), T(1.0), T(0.0)),  # y1=0 → flag=-2
                             (T(1.0), T(1.0), T(0.0), T(1.0))]
                    for (d1_0, d2_0, x1_0, y1_0) in cases
                        d1, d2, x1, y1, param = jcall_mg(handle, d1_0, d2_0, x1_0, y1_0)
                        # The H matrix encoded in `param` operates on the
                        # *unscaled* pair: H · (x1_orig, y1_orig) =
                        # (x1_new, 0). The d1/d2 scale factors are
                        # bookkeeping the caller tracks separately.
                        xv = T[x1_0]
                        yv = T[y1_0]
                        jcall_m(handle, 1, xv, 1, yv, 1, param)
                        @test isapprox(yv[1], zero(T);
                                       atol = atol * max(one(T), abs(x1)))
                        @test isapprox(xv[1], x1;
                                       atol = atol * max(one(T), abs(x1)))
                        # Full-rotation invariant: the scaled length is
                        # preserved across rotmg. Skip the y1=0 case where
                        # rotmg short-circuits to flag=-2 with d1, d2,
                        # x1 unchanged.
                        if d1_0 > 0 && d2_0 > 0 && y1_0 != 0 && d1 > 0
                            scaled_old = sqrt(d1_0)^2 * x1_0^2 + sqrt(d2_0)^2 * y1_0^2
                            scaled_new = sqrt(d1)^2 * x1^2
                            @test isapprox(scaled_new, scaled_old;
                                           atol = atol * max(one(T), scaled_old))
                        end
                    end
                end
            end
        end
    end
end

# (m, n, lda) matrix shapes. lda > m exercises non-contiguous columns —
# the routine must walk the leading dimension correctly and not touch
# padding rows.
const SHAPES_2D = [
    (1,    1,   1),
    (4,    4,   4),
    (8,    5,   8),
    (17,  13,  17),
    (17,  13,  20),   # lda > m
    (33,   1,  40),
    ( 1,  33,   1),
    (64,  64,  64),
]

# (incx, incy) pairs reused across gemv and ger.
const VEC_INCS = [(1, 1), (2, 3), (-1, 1), (1, -2)]

function ref_gemv(trans::Char, α, A, x, β, y0)
    op = trans == 'N' ? A : trans == 'T' ? transpose(A) : adjoint(A)
    return α .* (op * x) .+ β .* y0
end

function run_blas2_tests(handle, rng)
    @testset "BLAS Level 2" begin
        # ── gemv (y := α·op(A)·x + β·y) ──
        @testset "gemv" begin
            for (jcall, T, atol) in (
                    (sgemv, Float32,    1f-4),
                    (dgemv, Float64,    1e-11),
                    (cgemv, ComplexF32, 1f-4),
                    (zgemv, ComplexF64, 1e-11),
                )
                trset = T <: Complex ? CPLX_TR : REAL_TR
                @testset "$(nameof(jcall)) $T" begin
                    α = make_alpha(T)
                    β = make_beta(T)
                    for (m, n, lda) in SHAPES_2D, tr in trset, (incx, incy) in VEC_INCS
                        A_logical = randn(rng, T, m, n)
                        Astor = strided_matrix(A_logical, lda, rng)
                        Astor0 = copy(Astor)

                        xlen, ylen = (tr == 'N') ? (n, m) : (m, n)
                        x = rand_vec(T, xlen, rng)
                        y = rand_vec(T, ylen, rng)
                        xbuf = strided_buffer(x, incx)
                        ybuf = strided_buffer(y, incy)

                        jcall(handle, tr, m, n, α, Astor, lda,
                              xbuf, incx, β, ybuf, incy)

                        y_got = extract_strided(ybuf, ylen, incy)
                        y_ref = ref_gemv(tr, α, A_logical, x, β, y)
                        @test isapprox(y_got, y_ref;
                                       atol = atol * max(1, m * n))
                        # Padding rows (above logical m) untouched.
                        if lda > m
                            @test Astor[m+1:lda, :] == Astor0[m+1:lda, :]
                        end
                    end
                end
            end
        end

        # ── symv / hemv (A symmetric / Hermitian) ──
        @testset "symv / hemv" begin
            n_sweep = [1, 4, 17, 64]
            for (jcall, T, herm, atol) in (
                    (ssymv, Float32,    false, 1f-4),
                    (dsymv, Float64,    false, 1e-11),
                    (chemv, ComplexF32, true,  1f-4),
                    (zhemv, ComplexF64, true,  1e-11),
                )
                @testset "$(nameof(jcall)) $T" begin
                    α = make_alpha(T); β = make_beta(T)
                    for n in n_sweep, lda in (n, n + 3),
                        uplo in ('U', 'L'), (incx, incy) in VEC_INCS
                        # Random matrix with symmetrized / Hermitianised view
                        # for the reference. Pass the full random matrix into
                        # the routine — only the stored triangle is read.
                        A_logical = randn(rng, T, n, n)
                        if herm
                            for i in 1:n
                                A_logical[i, i] = real(A_logical[i, i])
                            end
                        end
                        Astor  = strided_matrix(A_logical, lda, rng)
                        Astor0 = copy(Astor)

                        x = rand_vec(T, n, rng)
                        y = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        ybuf = strided_buffer(y, incy)

                        jcall(handle, uplo, n, α, Astor, lda,
                              xbuf, incx, β, ybuf, incy)

                        Aview = if herm
                            Hermitian(A_logical, uplo == 'U' ? :U : :L)
                        else
                            Symmetric(A_logical, uplo == 'U' ? :U : :L)
                        end
                        y_ref = α .* (Aview * x) .+ β .* y
                        @test isapprox(extract_strided(ybuf, n, incy), y_ref;
                                       atol = atol * max(1, n * n))
                        @test Astor == Astor0  # routine must not write to A
                    end
                end
            end
        end

        # ── syr / her (rank-1 symmetric / Hermitian) ──
        @testset "syr / her" begin
            n_sweep = [1, 4, 17, 64]
            for (jcall, T, herm, αval, atol) in (
                    (ssyr, Float32,    false, 0.7f0,  1f-5),
                    (dsyr, Float64,    false, 0.7,    1e-12),
                    (cher, ComplexF32, true,  0.7f0,  1f-5),
                    (zher, ComplexF64, true,  0.7,    1e-12),
                )
                @testset "$(nameof(jcall)) $T" begin
                    for n in n_sweep, lda in (n, n + 3),
                        uplo in ('U', 'L'), incx in (1, 2, -1)
                        A_logical = randn(rng, T, n, n)
                        if herm
                            for i in 1:n
                                A_logical[i, i] = real(A_logical[i, i])
                            end
                        end
                        Astor  = strided_matrix(A_logical, lda, rng)
                        Astor0 = copy(Astor)

                        x = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)

                        jcall(handle, uplo, n, αval, xbuf, incx, Astor, lda)

                        # Build expected logical matrix in the stored triangle.
                        ker = herm ? αval .* (x * x') : αval .* (x * transpose(x))
                        A_expect = copy(A_logical)
                        for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n))
                            A_expect[i, j] += ker[i, j]
                            if herm && i == j
                                A_expect[i, j] = real(A_expect[i, j])
                            end
                        end
                        mask = uplo == 'U' ? triu(trues(n, n)) : tril(trues(n, n))
                        @test isapprox(extract_matrix(Astor, n, n)[mask],
                                       A_expect[mask]; atol = atol * max(1, n))
                        # Padding rows untouched.
                        if lda > n
                            @test Astor[n+1:lda, :] == Astor0[n+1:lda, :]
                        end
                    end
                end
            end
        end

        # ── syr2 / her2 (rank-2 symmetric / Hermitian) ──
        @testset "syr2 / her2" begin
            n_sweep = [1, 4, 17, 64]
            for (jcall, T, herm, atol) in (
                    (ssyr2, Float32,    false, 1f-5),
                    (dsyr2, Float64,    false, 1e-12),
                    (cher2, ComplexF32, true,  1f-5),
                    (zher2, ComplexF64, true,  1e-12),
                )
                @testset "$(nameof(jcall)) $T" begin
                    α = make_alpha(T)
                    for n in n_sweep, lda in (n, n + 3),
                        uplo in ('U', 'L'), (incx, incy) in VEC_INCS
                        A_logical = randn(rng, T, n, n)
                        if herm
                            for i in 1:n
                                A_logical[i, i] = real(A_logical[i, i])
                            end
                        end
                        Astor  = strided_matrix(A_logical, lda, rng)
                        Astor0 = copy(Astor)

                        x = rand_vec(T, n, rng)
                        y = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        ybuf = strided_buffer(y, incy)

                        jcall(handle, uplo, n, α, xbuf, incx, ybuf, incy, Astor, lda)

                        # Reference rank-2 update on the full logical matrix.
                        ker = herm ?
                            (α .* (x * y') .+ conj(α) .* (y * x')) :
                            (α .* (x * transpose(y)) .+ α .* (y * transpose(x)))
                        A_expect = copy(A_logical)
                        for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n))
                            A_expect[i, j] += ker[i, j]
                            if herm && i == j
                                A_expect[i, j] = real(A_expect[i, j])
                            end
                        end
                        mask = uplo == 'U' ? triu(trues(n, n)) : tril(trues(n, n))
                        @test isapprox(extract_matrix(Astor, n, n)[mask],
                                       A_expect[mask];
                                       atol = atol * max(1, n))
                        if lda > n
                            @test Astor[n+1:lda, :] == Astor0[n+1:lda, :]
                        end
                    end
                end
            end
        end

        # ── spmv / hpmv (symmetric / Hermitian packed) ──
        @testset "spmv / hpmv" begin
            n_sweep = [1, 4, 17]
            for (jcall, T, herm, atol) in (
                    (sspmv, Float32,    false, 1f-4),
                    (dspmv, Float64,    false, 1e-11),
                    (chpmv, ComplexF32, true,  1f-4),
                    (zhpmv, ComplexF64, true,  1e-11),
                )
                @testset "$(nameof(jcall)) $T" begin
                    α = make_alpha(T); β = make_beta(T)
                    for n in n_sweep, uplo in ('U', 'L'),
                        (incx, incy) in VEC_INCS
                        # Build random dense A; force Hermitian invariant.
                        A_dense = randn(rng, T, n, n)
                        if herm
                            for i in 1:n
                                A_dense[i, i] = real(A_dense[i, i])
                                for k in i+1:n
                                    A_dense[i, k] = conj(A_dense[k, i])
                                end
                            end
                        else
                            A_dense = (A_dense .+ transpose(A_dense)) ./ T(2)
                        end
                        AP = uplo == 'U' ? dense_to_packed_U(A_dense) :
                                            dense_to_packed_L(A_dense)
                        AP0 = copy(AP)

                        x = rand_vec(T, n, rng)
                        y = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        ybuf = strided_buffer(y, incy)

                        jcall(handle, uplo, n, α, AP, xbuf, incx, β, ybuf, incy)

                        Aview = herm ? Hermitian(A_dense, uplo == 'U' ? :U : :L) :
                                       Symmetric(A_dense, uplo == 'U' ? :U : :L)
                        y_ref = α .* (Aview * x) .+ β .* y
                        @test isapprox(extract_strided(ybuf, n, incy), y_ref;
                                       atol = atol * max(1, n * n))
                        @test AP == AP0  # routine must not write to AP
                    end
                end
            end
        end

        # ── spr / hpr (symmetric / Hermitian packed rank-1) ──
        @testset "spr / hpr" begin
            n_sweep = [1, 4, 17]
            for (jcall, T, herm, αval, atol) in (
                    (sspr, Float32,    false, 0.7f0, 1f-5),
                    (dspr, Float64,    false, 0.7,   1e-12),
                    (chpr, ComplexF32, true,  0.7f0, 1f-5),
                    (zhpr, ComplexF64, true,  0.7,   1e-12),
                )
                @testset "$(nameof(jcall)) $T" begin
                    for n in n_sweep, uplo in ('U', 'L'), incx in (1, 2, -1)
                        A_dense = randn(rng, T, n, n)
                        if herm
                            for i in 1:n
                                A_dense[i, i] = real(A_dense[i, i])
                            end
                        end
                        AP = uplo == 'U' ? dense_to_packed_U(A_dense) :
                                            dense_to_packed_L(A_dense)
                        x = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)

                        jcall(handle, uplo, n, αval, xbuf, incx, AP)

                        # Reference: build expected packed via dense update.
                        ker = herm ? αval .* (x * x') : αval .* (x * transpose(x))
                        A_expect = copy(A_dense)
                        for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n))
                            A_expect[i, j] += ker[i, j]
                            if herm && i == j
                                A_expect[i, j] = real(A_expect[i, j])
                            end
                        end
                        AP_expect = uplo == 'U' ? dense_to_packed_U(A_expect) :
                                                   dense_to_packed_L(A_expect)
                        @test isapprox(AP, AP_expect; atol = atol * max(1, n))
                    end
                end
            end
        end

        # ── spr2 / hpr2 (symmetric / Hermitian packed rank-2) ──
        @testset "spr2 / hpr2" begin
            n_sweep = [1, 4, 17]
            for (jcall, T, herm, atol) in (
                    (sspr2, Float32,    false, 1f-5),
                    (dspr2, Float64,    false, 1e-12),
                    (chpr2, ComplexF32, true,  1f-5),
                    (zhpr2, ComplexF64, true,  1e-12),
                )
                @testset "$(nameof(jcall)) $T" begin
                    α = make_alpha(T)
                    for n in n_sweep, uplo in ('U', 'L'), (incx, incy) in VEC_INCS
                        A_dense = randn(rng, T, n, n)
                        if herm
                            for i in 1:n
                                A_dense[i, i] = real(A_dense[i, i])
                            end
                        end
                        AP = uplo == 'U' ? dense_to_packed_U(A_dense) :
                                            dense_to_packed_L(A_dense)
                        x = rand_vec(T, n, rng)
                        y = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        ybuf = strided_buffer(y, incy)

                        jcall(handle, uplo, n, α, xbuf, incx, ybuf, incy, AP)

                        ker = herm ?
                            (α .* (x * y') .+ conj(α) .* (y * x')) :
                            (α .* (x * transpose(y)) .+ α .* (y * transpose(x)))
                        A_expect = copy(A_dense)
                        for j in 1:n, i in (uplo == 'U' ? (1:j) : (j:n))
                            A_expect[i, j] += ker[i, j]
                            if herm && i == j
                                A_expect[i, j] = real(A_expect[i, j])
                            end
                        end
                        AP_expect = uplo == 'U' ? dense_to_packed_U(A_expect) :
                                                   dense_to_packed_L(A_expect)
                        @test isapprox(AP, AP_expect; atol = atol * max(1, n))
                    end
                end
            end
        end

        # ── tpmv / tpsv (triangular packed) ──
        @testset "tpmv / tpsv" begin
            n_sweep = [1, 5, 16]
            for (jcall_mv, jcall_sv, T, atol_mv, atol_sv) in (
                    (stpmv, stpsv, Float32,    1f-4, 1f-3),
                    (dtpmv, dtpsv, Float64,    1e-11, 1e-10),
                    (ctpmv, ctpsv, ComplexF32, 1f-4, 1f-3),
                    (ztpmv, ztpsv, ComplexF64, 1e-11, 1e-10),
                )
                trset = T <: Complex ? CPLX_TR : REAL_TR
                @testset "$(nameof(jcall_mv)) / $(nameof(jcall_sv)) $T" begin
                    for n in n_sweep, uplo in ('U', 'L'),
                        trans in trset, diag_c in ('N', 'U'), incx in (1, 2, -1)
                        # Diagonally-dominant random tri matrix → stable tpsv.
                        A_dense = randn(rng, T, n, n) ./ T(n) .+
                                  Matrix{T}(T(1) * I, n, n)
                        AP = uplo == 'U' ? dense_to_packed_U(A_dense) :
                                            dense_to_packed_L(A_dense)
                        AP0 = copy(AP)

                        Atri = if uplo == 'U'
                            diag_c == 'U' ? UnitUpperTriangular(A_dense) :
                                            UpperTriangular(A_dense)
                        else
                            diag_c == 'U' ? UnitLowerTriangular(A_dense) :
                                            LowerTriangular(A_dense)
                        end
                        Aop = trans == 'N' ? Atri :
                              trans == 'T' ? transpose(Atri) : adjoint(Atri)

                        # tpmv
                        x = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        jcall_mv(handle, uplo, trans, diag_c, n, AP, xbuf, incx)
                        @test isapprox(extract_strided(xbuf, n, incx), Aop * x;
                                       atol = atol_mv * max(1, n))
                        @test AP == AP0  # AP unchanged

                        # tpsv
                        b = rand_vec(T, n, rng)
                        bbuf = strided_buffer(b, incx)
                        jcall_sv(handle, uplo, trans, diag_c, n, AP, bbuf, incx)
                        @test isapprox(extract_strided(bbuf, n, incx), Aop \ b;
                                       atol = atol_sv * max(1, n))
                        @test AP == AP0
                    end
                end
            end
        end

        # ── gbmv (general band) ──
        @testset "gbmv" begin
            n_sweep = [1, 5, 16]
            band_sweep = [(0,0), (1,0), (0,1), (1,2), (2,3)]
            for (jcall, T, atol) in (
                    (sgbmv, Float32,    1f-4),
                    (dgbmv, Float64,    1e-11),
                    (cgbmv, ComplexF32, 1f-4),
                    (zgbmv, ComplexF64, 1e-11),
                )
                trset = T <: Complex ? CPLX_TR : REAL_TR
                @testset "$(nameof(jcall)) $T" begin
                    α = make_alpha(T); β = make_beta(T)
                    for n in n_sweep, m in (n, n + 2),
                        (kl, ku) in band_sweep, trans in trset,
                        (incx, incy) in VEC_INCS
                        kl_eff = min(kl, m - 1 < 0 ? 0 : m - 1)
                        ku_eff = min(ku, n - 1 < 0 ? 0 : n - 1)
                        lda = kl_eff + ku_eff + 1
                        A_dense = rand_band_dense(T, m, n, kl_eff, ku_eff, rng)
                        AB = dense_to_gb(A_dense, kl_eff, ku_eff, lda)

                        xlen, ylen = (trans == 'N') ? (n, m) : (m, n)
                        x = rand_vec(T, xlen, rng)
                        y = rand_vec(T, ylen, rng)
                        xbuf = strided_buffer(x, incx)
                        ybuf = strided_buffer(y, incy)

                        jcall(handle, trans, m, n, kl_eff, ku_eff, α,
                              AB, lda, xbuf, incx, β, ybuf, incy)

                        op = trans == 'N' ? A_dense :
                             trans == 'T' ? transpose(A_dense) : adjoint(A_dense)
                        y_ref = α .* (op * x) .+ β .* y
                        @test isapprox(extract_strided(ybuf, ylen, incy), y_ref;
                                       atol = atol * max(1, m * n))
                    end
                end
            end
        end

        # ── sbmv / hbmv (symmetric / Hermitian band) ──
        @testset "sbmv / hbmv" begin
            n_sweep = [1, 5, 16]
            for (jcall, T, herm, atol) in (
                    (ssbmv, Float32,    false, 1f-4),
                    (dsbmv, Float64,    false, 1e-11),
                    (chbmv, ComplexF32, true,  1f-4),
                    (zhbmv, ComplexF64, true,  1e-11),
                )
                @testset "$(nameof(jcall)) $T" begin
                    α = make_alpha(T); β = make_beta(T)
                    for n in n_sweep, k in 0:min(3, n - 1),
                        uplo in ('U', 'L'), (incx, incy) in VEC_INCS
                        lda = k + 1
                        A_dense = herm ? rand_herm_band_dense(T, n, k, rng) :
                                          rand_sym_band_dense(T, n, k, rng)
                        AB = uplo == 'U' ? dense_to_band_U(A_dense, k, lda) :
                                            dense_to_band_L(A_dense, k, lda)
                        x = rand_vec(T, n, rng)
                        y = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        ybuf = strided_buffer(y, incy)

                        jcall(handle, uplo, n, k, α, AB, lda,
                              xbuf, incx, β, ybuf, incy)

                        y_ref = α .* (A_dense * x) .+ β .* y
                        @test isapprox(extract_strided(ybuf, n, incy), y_ref;
                                       atol = atol * max(1, n * (k + 1)))
                    end
                end
            end
        end

        # ── tbmv / tbsv (triangular band) ──
        @testset "tbmv / tbsv" begin
            n_sweep = [1, 5, 16]
            for (jcall_mv, jcall_sv, T, atol_mv, atol_sv) in (
                    (stbmv, stbsv, Float32,    1f-4, 1f-3),
                    (dtbmv, dtbsv, Float64,    1e-11, 1e-10),
                    (ctbmv, ctbsv, ComplexF32, 1f-4, 1f-3),
                    (ztbmv, ztbsv, ComplexF64, 1e-11, 1e-10),
                )
                trset = T <: Complex ? CPLX_TR : REAL_TR
                @testset "$(nameof(jcall_mv)) / $(nameof(jcall_sv)) $T" begin
                    for n in n_sweep, k in 0:min(3, n - 1),
                        uplo in ('U', 'L'), trans in trset,
                        diag_c in ('N', 'U'), incx in (1, 2, -1)
                        lda = k + 1
                        A_dense = rand_tri_band_dense(T, n, k, uplo, rng)
                        AB = uplo == 'U' ? dense_to_band_U(A_dense, k, lda) :
                                            dense_to_band_L(A_dense, k, lda)
                        Atri = if uplo == 'U'
                            diag_c == 'U' ? UnitUpperTriangular(A_dense) :
                                            UpperTriangular(A_dense)
                        else
                            diag_c == 'U' ? UnitLowerTriangular(A_dense) :
                                            LowerTriangular(A_dense)
                        end
                        Aop = trans == 'N' ? Atri :
                              trans == 'T' ? transpose(Atri) : adjoint(Atri)

                        # tbmv
                        x = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        jcall_mv(handle, uplo, trans, diag_c, n, k, AB, lda, xbuf, incx)
                        @test isapprox(extract_strided(xbuf, n, incx), Aop * x;
                                       atol = atol_mv * max(1, n * (k + 1)))

                        # tbsv
                        b = rand_vec(T, n, rng)
                        bbuf = strided_buffer(b, incx)
                        jcall_sv(handle, uplo, trans, diag_c, n, k, AB, lda, bbuf, incx)
                        @test isapprox(extract_strided(bbuf, n, incx), Aop \ b;
                                       atol = atol_sv * max(1, n * (k + 1)))
                    end
                end
            end
        end

        # ── trmv / trsv (triangular matrix-vector / triangular solve) ──
        @testset "trmv / trsv" begin
            n_sweep = [1, 5, 16]
            # For trsv numerical stability, build A as I + 0.1·random; for
            # trmv any matrix works.
            for (jcall_mv, jcall_sv, T, atol_mv, atol_sv) in (
                    (strmv, strsv, Float32,    1f-4, 1f-3),
                    (dtrmv, dtrsv, Float64,    1e-11, 1e-10),
                    (ctrmv, ctrsv, ComplexF32, 1f-4, 1f-3),
                    (ztrmv, ztrsv, ComplexF64, 1e-11, 1e-10),
                )
                trset = T <: Complex ? CPLX_TR : REAL_TR
                @testset "$(nameof(jcall_mv)) / $(nameof(jcall_sv)) $T" begin
                    for n in n_sweep, lda in (n, n + 2),
                        uplo in ('U', 'L'), trans in trset,
                        diag_c in ('N', 'U'), incx in (1, 2, -1)

                        # Build a well-conditioned A (diagonally dominant) so
                        # trsv stays numerically stable.
                        A_logical = randn(rng, T, n, n) ./ T(n) .+
                                    Matrix{T}(T(1) * I, n, n)
                        Astor  = strided_matrix(A_logical, lda, rng)
                        Astor0 = copy(Astor)

                        # Build the Julia triangular view that the routine
                        # implicitly sees.
                        Atri = if uplo == 'U'
                            diag_c == 'U' ? UnitUpperTriangular(A_logical) :
                                            UpperTriangular(A_logical)
                        else
                            diag_c == 'U' ? UnitLowerTriangular(A_logical) :
                                            LowerTriangular(A_logical)
                        end
                        Aop = trans == 'N' ? Atri :
                              trans == 'T' ? transpose(Atri) :
                                             adjoint(Atri)

                        # ── trmv ──
                        x = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        jcall_mv(handle, uplo, trans, diag_c, n, Astor, lda, xbuf, incx)
                        @test isapprox(extract_strided(xbuf, n, incx), Aop * x;
                                       atol = atol_mv * max(1, n))
                        @test Astor == Astor0  # A unchanged

                        # ── trsv ──
                        b = rand_vec(T, n, rng)
                        bbuf = strided_buffer(b, incx)
                        jcall_sv(handle, uplo, trans, diag_c, n, Astor, lda, bbuf, incx)
                        @test isapprox(extract_strided(bbuf, n, incx), Aop \ b;
                                       atol = atol_sv * max(1, n))
                        @test Astor == Astor0
                        if lda > n
                            @test Astor[n+1:lda, :] == Astor0[n+1:lda, :]
                        end
                    end
                end
            end
        end

        # ── ger / geru / gerc (A := α·x·op(y) + A) ──
        @testset "ger / geru / gerc" begin
            for (jcall, T, conj_y, atol) in (
                    (sger,  Float32,    false, 1f-5),
                    (dger,  Float64,    false, 1e-12),
                    (cgeru, ComplexF32, false, 1f-5),
                    (zgeru, ComplexF64, false, 1e-12),
                    (cgerc, ComplexF32, true,  1f-5),
                    (zgerc, ComplexF64, true,  1e-12),
                )
                @testset "$(nameof(jcall)) $T" begin
                    α = make_alpha(T)
                    for (m, n, lda) in SHAPES_2D, (incx, incy) in VEC_INCS
                        A_logical = randn(rng, T, m, n)
                        Astor  = strided_matrix(A_logical, lda, rng)
                        Astor0 = copy(Astor)

                        x = rand_vec(T, m, rng)
                        y = rand_vec(T, n, rng)
                        xbuf = strided_buffer(x, incx)
                        ybuf = strided_buffer(y, incy)

                        jcall(handle, m, n, α, xbuf, incx, ybuf, incy, Astor, lda)

                        yop = conj_y ? conj.(y) : y
                        A_ref = A_logical .+ α .* (x * transpose(yop))
                        @test isapprox(extract_matrix(Astor, m, n), A_ref;
                                       atol = atol * max(1, m * n))
                        if lda > m
                            @test Astor[m+1:lda, :] == Astor0[m+1:lda, :]
                        end
                    end
                end
            end
        end
    end
end

function run_tests(handle)
    rng = MersenneTwister(0x4a55424c)  # "JUBL"

    @testset "JuBLAS libjublas.$(Libdl.dlext)" begin
        for (jcall, T, trset) in (
                (dgemm_jublas!, Float64,     REAL_TR),
                (sgemm_jublas!, Float32,     REAL_TR),
                (zgemm_jublas!, ComplexF64,  CPLX_TR),
                (cgemm_jublas!, ComplexF32,  CPLX_TR),
            )
            atol = real(T) === Float32 ? 1f-3 : 1e-9
            @testset "$(nameof(jcall)) $T" begin
                for (M, N, K) in SHAPES, tA in trset, tB in trset
                    A, B = make_inputs(T, tA, tB, M, N, K, rng)
                    C0   = randmat(T, M, N, rng)
                    α    = T(0.7)
                    β    = T(-0.3)
                    Cref = ref_gemm(tA, tB, α, A, B, β, C0)
                    Cgot = copy(C0)
                    jcall(handle, tA, tB, α, A, B, β, Cgot)
                    @test isapprox(Cgot, Cref; atol = atol * max(1, M * K))
                end
            end
        end

        @testset "stub symbols are exported" begin
            # Just resolve the symbols — calling them aborts the process.
            for sym in (:dsymm_, :zhemm_, :ssyrk_, :zherk_, :dsyr2k_,
                        :cher2k_, :strmm_, :ztrsm_)
                @test Libdl.dlsym_e(handle, sym) != C_NULL
            end
        end

        run_blas1_tests(handle, rng)
        run_blas2_tests(handle, rng)
    end
end

function main()
    explicit = isempty(ARGS) ? nothing : ARGS[1]
    libpath  = locate_library(explicit)
    if libpath === nothing
        @info """skipping libjublas (juliac AOT) tests — no library found.
                 Searched: $(join(SEARCH_PATHS, ", "))
                 Build first with juliac (`--privatize` is required), or set JUBLAS_LIB=/path/to/lib."""
        return
    end
    @info "loading libjublas" libpath
    handle = open_library(libpath)
    try
        run_tests(handle)
    finally
        Libdl.dlclose(handle)
    end
end

isinteractive() || main()
