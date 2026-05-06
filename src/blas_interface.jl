# Fortran-ABI Level 3 BLAS entry points for AOT compilation via juliac.
#
# LP64 convention: BlasInt = Cint (32-bit). All arguments are passed by
# reference (Ptr) — the standard Fortran calling convention. Trailing hidden
# Csize_t length arguments follow the gfortran convention for character
# string parameters; we accept and ignore them.
#
# Only GEMM (sgemm_/dgemm_/cgemm_/zgemm_) is wired through to the real
# kernels. SYMM/HEMM/SYRK/HERK/SYR2K/HER2K/TRMM/TRSM are stubs that print
# a message and abort — fill them in once the corresponding kernels exist.

# `BlasInt` is defined in blas1.jl, which is included first.

@inline _isN(c::UInt8) = (c == UInt8('N')) | (c == UInt8('n'))
@inline _isT(c::UInt8) = (c == UInt8('T')) | (c == UInt8('t'))

# ─── GEMM ──────────────────────────────────────────────────────────────────
#
# Trim path bypasses the kwarg trampoline `gemm!` in `gemm.jl` and dispatches
# straight into `_gemm!` with a concretely-typed kernel. Going through the
# kwarg machinery causes trim=safe to widen the `default_kernel(T, ...)`
# result to `SIMDKernel{W,MR,NR,T} where {...}` (UnionAll) at the boundary
# of `var"#gemm!#2"`, which it then refuses to compile. Inlining the
# kernel-narrowing isa-chain here keeps every dispatched call concrete.

@inline function _kernel_dispatch!(C::AbstractMatrix{T}, A, B, α, β) where {T}
    k = default_kernel(T, size(C,1), size(C,2), size(A,2))
    # Real kernels.
    if     k isa SIMDKernel{8, 24,  8, Float64}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    elseif k isa SIMDKernel{8, 16, 14, Float64}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    elseif k isa SIMDKernel{4,  8,  6, Float64}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    elseif k isa SIMDKernel{2,  2,  4, Float64}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    elseif k isa SIMDKernel{16, 32, 12, Float32}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    elseif k isa SIMDKernel{8,  16,  6, Float32}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    elseif k isa SIMDKernel{4,   4,  6, Float32}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    # Complex kernels.
    elseif k isa SIMDKernel{8,   8,  8, ComplexF64}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    elseif k isa SIMDKernel{4,   4,  6, ComplexF64}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    elseif k isa SIMDKernel{2,   2,  4, ComplexF64}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    elseif k isa SIMDKernel{16, 32,  6, ComplexF32}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    elseif k isa SIMDKernel{8,   8,  6, ComplexF32}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    elseif k isa SIMDKernel{4,   4,  6, ComplexF32}
        _gemm!(C, A, B, α, β, k, nothing, nothing)
    end
    return nothing
end

@inline function _gemm_with_A!(C::AbstractMatrix{T}, Aop,
                                B_ptr::Ptr{T}, ldb::Int, transB::UInt8,
                                K::Int, N::Int, α::T, β::T) where {T}
    if _isN(transB)
        Barr = unsafe_wrap(Array, B_ptr, (ldb, N))
        _kernel_dispatch!(C, Aop, view(Barr, 1:K, 1:N), α, β)
    elseif _isT(transB)
        Barr = unsafe_wrap(Array, B_ptr, (ldb, K))
        _kernel_dispatch!(C, Aop, transpose(view(Barr, 1:N, 1:K)), α, β)
    else
        Barr = unsafe_wrap(Array, B_ptr, (ldb, K))
        _kernel_dispatch!(C, Aop, adjoint(view(Barr, 1:N, 1:K)), α, β)
    end
    return nothing
end

@inline function _gemm_dispatch!(C::AbstractMatrix{T},
                                  A_ptr::Ptr{T}, lda::Int, transA::UInt8,
                                  B_ptr::Ptr{T}, ldb::Int, transB::UInt8,
                                  M::Int, N::Int, K::Int,
                                  α::T, β::T) where {T}
    if _isN(transA)
        Aarr = unsafe_wrap(Array, A_ptr, (lda, K))
        Av = view(Aarr, 1:M, 1:K)
        _gemm_with_A!(C, Av, B_ptr, ldb, transB, K, N, α, β)
    elseif _isT(transA)
        Aarr = unsafe_wrap(Array, A_ptr, (lda, M))
        Av = transpose(view(Aarr, 1:K, 1:M))
        _gemm_with_A!(C, Av, B_ptr, ldb, transB, K, N, α, β)
    else
        Aarr = unsafe_wrap(Array, A_ptr, (lda, M))
        Av = adjoint(view(Aarr, 1:K, 1:M))
        _gemm_with_A!(C, Av, B_ptr, ldb, transB, K, N, α, β)
    end
    return nothing
end

@inline function _gemm_entry!(::Type{T},
        transA::Ptr{UInt8}, transB::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt}, k::Ptr{BlasInt},
        α::Ptr{T}, A::Ptr{T}, lda::Ptr{BlasInt},
        B::Ptr{T}, ldb::Ptr{BlasInt},
        β::Ptr{T}, C::Ptr{T}, ldc::Ptr{BlasInt}) where {T}
    M   = Int(unsafe_load(m))
    N   = Int(unsafe_load(n))
    K   = Int(unsafe_load(k))
    LDA = Int(unsafe_load(lda))
    LDB = Int(unsafe_load(ldb))
    LDC = Int(unsafe_load(ldc))
    αv  = unsafe_load(α)
    βv  = unsafe_load(β)
    tA  = unsafe_load(transA)
    tB  = unsafe_load(transB)
    Carr = unsafe_wrap(Array, C, (LDC, N))
    Cv = view(Carr, 1:M, 1:N)
    _gemm_dispatch!(Cv, A, LDA, tA, B, LDB, tB, M, N, K, αv, βv)
    return nothing
end

Base.@ccallable function sgemm_(transA::Ptr{UInt8}, transB::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{Float32}, A::Ptr{Float32}, lda::Ptr{BlasInt},
        B::Ptr{Float32}, ldb::Ptr{BlasInt},
        beta::Ptr{Float32}, C::Ptr{Float32}, ldc::Ptr{BlasInt},
        _la::Csize_t, _lb::Csize_t)::Cvoid
    _gemm_entry!(Float32, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    return
end

Base.@ccallable function dgemm_(transA::Ptr{UInt8}, transB::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{Float64}, A::Ptr{Float64}, lda::Ptr{BlasInt},
        B::Ptr{Float64}, ldb::Ptr{BlasInt},
        beta::Ptr{Float64}, C::Ptr{Float64}, ldc::Ptr{BlasInt},
        _la::Csize_t, _lb::Csize_t)::Cvoid
    _gemm_entry!(Float64, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    return
end

Base.@ccallable function cgemm_(transA::Ptr{UInt8}, transB::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF32}, ldb::Ptr{BlasInt},
        beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Ptr{BlasInt},
        _la::Csize_t, _lb::Csize_t)::Cvoid
    _gemm_entry!(ComplexF32, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    return
end

Base.@ccallable function zgemm_(transA::Ptr{UInt8}, transB::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{ComplexF64}, A::Ptr{ComplexF64}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF64}, ldb::Ptr{BlasInt},
        beta::Ptr{ComplexF64}, C::Ptr{ComplexF64}, ldc::Ptr{BlasInt},
        _la::Csize_t, _lb::Csize_t)::Cvoid
    _gemm_entry!(ComplexF64, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    return
end

# ─── Stubs for the rest of Level 3 ─────────────────────────────────────────
#
# Each stub prints a one-line diagnostic via libc puts and aborts, so calls
# into the not-yet-implemented routines fail loudly rather than silently
# returning garbage. Replace the body with a real wrapper once the kernel
# lands.

@noinline function _unimpl(msg::String)
    @ccall puts(msg::Cstring)::Cint
    @ccall abort()::Cvoid
    return
end

# SYMM: C := α·A·B + β·C  or  α·B·A + β·C   (A symmetric)
# args: (side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc) + 2 hidden lens

Base.@ccallable function ssymm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{Float32}, A::Ptr{Float32}, lda::Ptr{BlasInt},
        B::Ptr{Float32}, ldb::Ptr{BlasInt},
        beta::Ptr{Float32}, C::Ptr{Float32}, ldc::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t)::Cvoid
    _unimpl("JuBLAS: ssymm_ not implemented")
    return
end

Base.@ccallable function dsymm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{Float64}, A::Ptr{Float64}, lda::Ptr{BlasInt},
        B::Ptr{Float64}, ldb::Ptr{BlasInt},
        beta::Ptr{Float64}, C::Ptr{Float64}, ldc::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t)::Cvoid
    _unimpl("JuBLAS: dsymm_ not implemented")
    return
end

Base.@ccallable function csymm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF32}, ldb::Ptr{BlasInt},
        beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t)::Cvoid
    _unimpl("JuBLAS: csymm_ not implemented")
    return
end

Base.@ccallable function zsymm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{ComplexF64}, A::Ptr{ComplexF64}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF64}, ldb::Ptr{BlasInt},
        beta::Ptr{ComplexF64}, C::Ptr{ComplexF64}, ldc::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t)::Cvoid
    _unimpl("JuBLAS: zsymm_ not implemented")
    return
end

# HEMM: A Hermitian (complex only).
Base.@ccallable function chemm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF32}, ldb::Ptr{BlasInt},
        beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t)::Cvoid
    _unimpl("JuBLAS: chemm_ not implemented")
    return
end

Base.@ccallable function zhemm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{ComplexF64}, A::Ptr{ComplexF64}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF64}, ldb::Ptr{BlasInt},
        beta::Ptr{ComplexF64}, C::Ptr{ComplexF64}, ldc::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t)::Cvoid
    _unimpl("JuBLAS: zhemm_ not implemented")
    return
end

# SYRK: C := α·A·Aᵀ + β·C  (or α·Aᵀ·A + β·C)
# args: (uplo, trans, n, k, alpha, A, lda, beta, C, ldc) + 2 hidden lens

Base.@ccallable function ssyrk_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{Float32}, A::Ptr{Float32}, lda::Ptr{BlasInt},
        beta::Ptr{Float32}, C::Ptr{Float32}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: ssyrk_ not implemented")
    return
end

Base.@ccallable function dsyrk_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{Float64}, A::Ptr{Float64}, lda::Ptr{BlasInt},
        beta::Ptr{Float64}, C::Ptr{Float64}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: dsyrk_ not implemented")
    return
end

Base.@ccallable function csyrk_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Ptr{BlasInt},
        beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: csyrk_ not implemented")
    return
end

Base.@ccallable function zsyrk_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{ComplexF64}, A::Ptr{ComplexF64}, lda::Ptr{BlasInt},
        beta::Ptr{ComplexF64}, C::Ptr{ComplexF64}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: zsyrk_ not implemented")
    return
end

# HERK: alpha and beta are REAL even when matrices are complex.
Base.@ccallable function cherk_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{Float32}, A::Ptr{ComplexF32}, lda::Ptr{BlasInt},
        beta::Ptr{Float32}, C::Ptr{ComplexF32}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: cherk_ not implemented")
    return
end

Base.@ccallable function zherk_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{Float64}, A::Ptr{ComplexF64}, lda::Ptr{BlasInt},
        beta::Ptr{Float64}, C::Ptr{ComplexF64}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: zherk_ not implemented")
    return
end

# SYR2K: C := α·A·Bᵀ + α·B·Aᵀ + β·C
# args: (uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc) + 2 hidden lens

Base.@ccallable function ssyr2k_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{Float32}, A::Ptr{Float32}, lda::Ptr{BlasInt},
        B::Ptr{Float32}, ldb::Ptr{BlasInt},
        beta::Ptr{Float32}, C::Ptr{Float32}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: ssyr2k_ not implemented")
    return
end

Base.@ccallable function dsyr2k_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{Float64}, A::Ptr{Float64}, lda::Ptr{BlasInt},
        B::Ptr{Float64}, ldb::Ptr{BlasInt},
        beta::Ptr{Float64}, C::Ptr{Float64}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: dsyr2k_ not implemented")
    return
end

Base.@ccallable function csyr2k_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF32}, ldb::Ptr{BlasInt},
        beta::Ptr{ComplexF32}, C::Ptr{ComplexF32}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: csyr2k_ not implemented")
    return
end

Base.@ccallable function zsyr2k_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{ComplexF64}, A::Ptr{ComplexF64}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF64}, ldb::Ptr{BlasInt},
        beta::Ptr{ComplexF64}, C::Ptr{ComplexF64}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: zsyr2k_ not implemented")
    return
end

# HER2K: beta is REAL.
Base.@ccallable function cher2k_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF32}, ldb::Ptr{BlasInt},
        beta::Ptr{Float32}, C::Ptr{ComplexF32}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: cher2k_ not implemented")
    return
end

Base.@ccallable function zher2k_(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
        n::Ptr{BlasInt}, k::Ptr{BlasInt},
        alpha::Ptr{ComplexF64}, A::Ptr{ComplexF64}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF64}, ldb::Ptr{BlasInt},
        beta::Ptr{Float64}, C::Ptr{ComplexF64}, ldc::Ptr{BlasInt},
        _lu::Csize_t, _lt::Csize_t)::Cvoid
    _unimpl("JuBLAS: zher2k_ not implemented")
    return
end

# TRMM: B := α·op(A)·B  or  α·B·op(A)   (A triangular)
# args: (side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb) + 4 hidden lens

Base.@ccallable function strmm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        transA::Ptr{UInt8}, diag::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{Float32}, A::Ptr{Float32}, lda::Ptr{BlasInt},
        B::Ptr{Float32}, ldb::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t, _lt::Csize_t, _ld::Csize_t)::Cvoid
    _unimpl("JuBLAS: strmm_ not implemented")
    return
end

Base.@ccallable function dtrmm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        transA::Ptr{UInt8}, diag::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{Float64}, A::Ptr{Float64}, lda::Ptr{BlasInt},
        B::Ptr{Float64}, ldb::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t, _lt::Csize_t, _ld::Csize_t)::Cvoid
    _unimpl("JuBLAS: dtrmm_ not implemented")
    return
end

Base.@ccallable function ctrmm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        transA::Ptr{UInt8}, diag::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF32}, ldb::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t, _lt::Csize_t, _ld::Csize_t)::Cvoid
    _unimpl("JuBLAS: ctrmm_ not implemented")
    return
end

Base.@ccallable function ztrmm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        transA::Ptr{UInt8}, diag::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{ComplexF64}, A::Ptr{ComplexF64}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF64}, ldb::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t, _lt::Csize_t, _ld::Csize_t)::Cvoid
    _unimpl("JuBLAS: ztrmm_ not implemented")
    return
end

# TRSM: solve op(A)·X = α·B  or  X·op(A) = α·B   (A triangular)
# Same arg list as TRMM.

Base.@ccallable function strsm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        transA::Ptr{UInt8}, diag::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{Float32}, A::Ptr{Float32}, lda::Ptr{BlasInt},
        B::Ptr{Float32}, ldb::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t, _lt::Csize_t, _ld::Csize_t)::Cvoid
    _unimpl("JuBLAS: strsm_ not implemented")
    return
end

Base.@ccallable function dtrsm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        transA::Ptr{UInt8}, diag::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{Float64}, A::Ptr{Float64}, lda::Ptr{BlasInt},
        B::Ptr{Float64}, ldb::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t, _lt::Csize_t, _ld::Csize_t)::Cvoid
    _unimpl("JuBLAS: dtrsm_ not implemented")
    return
end

Base.@ccallable function ctrsm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        transA::Ptr{UInt8}, diag::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{ComplexF32}, A::Ptr{ComplexF32}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF32}, ldb::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t, _lt::Csize_t, _ld::Csize_t)::Cvoid
    _unimpl("JuBLAS: ctrsm_ not implemented")
    return
end

Base.@ccallable function ztrsm_(side::Ptr{UInt8}, uplo::Ptr{UInt8},
        transA::Ptr{UInt8}, diag::Ptr{UInt8},
        m::Ptr{BlasInt}, n::Ptr{BlasInt},
        alpha::Ptr{ComplexF64}, A::Ptr{ComplexF64}, lda::Ptr{BlasInt},
        B::Ptr{ComplexF64}, ldb::Ptr{BlasInt},
        _ls::Csize_t, _lu::Csize_t, _lt::Csize_t, _ld::Csize_t)::Cvoid
    _unimpl("JuBLAS: ztrsm_ not implemented")
    return
end

# ─── Level 1 BLAS ─────────────────────────────────────────────────────────
#
# All scalars passed by reference; no character-string args, so no trailing
# hidden length parameters. Generated with @eval over (suffix, T) tuples
# to keep the source legible — every @ccallable below expands to exactly
# the same straight-line wrapper pattern.

# axpy_, copy_, swap_  — same signature, eltype-parameterised body.
for (suffix, T) in ((:s, Float32), (:d, Float64),
                    (:c, ComplexF32), (:z, ComplexF64))
    sym_axpy = Symbol(suffix, "axpy_")
    sym_copy = Symbol(suffix, "copy_")
    sym_swap = Symbol(suffix, "swap_")
    @eval begin
        Base.@ccallable function $sym_axpy(n::Ptr{BlasInt}, α::Ptr{$T},
                x::Ptr{$T}, incx::Ptr{BlasInt},
                y::Ptr{$T}, incy::Ptr{BlasInt})::Cvoid
            _axpy_impl!(Int(unsafe_load(n)), unsafe_load(α),
                        x, Int(unsafe_load(incx)),
                        y, Int(unsafe_load(incy)))
            return
        end

        Base.@ccallable function $sym_copy(n::Ptr{BlasInt},
                x::Ptr{$T}, incx::Ptr{BlasInt},
                y::Ptr{$T}, incy::Ptr{BlasInt})::Cvoid
            _copy_impl!(Int(unsafe_load(n)),
                        x, Int(unsafe_load(incx)),
                        y, Int(unsafe_load(incy)))
            return
        end

        Base.@ccallable function $sym_swap(n::Ptr{BlasInt},
                x::Ptr{$T}, incx::Ptr{BlasInt},
                y::Ptr{$T}, incy::Ptr{BlasInt})::Cvoid
            _swap_impl!(Int(unsafe_load(n)),
                        x, Int(unsafe_load(incx)),
                        y, Int(unsafe_load(incy)))
            return
        end
    end
end

# scal_  — α and x same eltype.
for (suffix, T) in ((:s, Float32), (:d, Float64),
                    (:c, ComplexF32), (:z, ComplexF64))
    sym = Symbol(suffix, "scal_")
    @eval Base.@ccallable function $sym(n::Ptr{BlasInt}, α::Ptr{$T},
            x::Ptr{$T}, incx::Ptr{BlasInt})::Cvoid
        _scal_impl!(Int(unsafe_load(n)), unsafe_load(α),
                    x, Int(unsafe_load(incx)))
        return
    end
end

# Mixed-precision scal: real α, complex x.
Base.@ccallable function csscal_(n::Ptr{BlasInt}, α::Ptr{Float32},
        x::Ptr{ComplexF32}, incx::Ptr{BlasInt})::Cvoid
    _scal_impl!(Int(unsafe_load(n)), unsafe_load(α),
                x, Int(unsafe_load(incx)))
    return
end

Base.@ccallable function zdscal_(n::Ptr{BlasInt}, α::Ptr{Float64},
        x::Ptr{ComplexF64}, incx::Ptr{BlasInt})::Cvoid
    _scal_impl!(Int(unsafe_load(n)), unsafe_load(α),
                x, Int(unsafe_load(incx)))
    return
end

# Real dot — sdot_, ddot_ return the dot product as a value.
Base.@ccallable function sdot_(n::Ptr{BlasInt},
        x::Ptr{Float32}, incx::Ptr{BlasInt},
        y::Ptr{Float32}, incy::Ptr{BlasInt})::Float32
    return _dot_impl(Float32, Int(unsafe_load(n)),
                     x, Int(unsafe_load(incx)),
                     y, Int(unsafe_load(incy)))
end

Base.@ccallable function ddot_(n::Ptr{BlasInt},
        x::Ptr{Float64}, incx::Ptr{BlasInt},
        y::Ptr{Float64}, incy::Ptr{BlasInt})::Float64
    return _dot_impl(Float64, Int(unsafe_load(n)),
                     x, Int(unsafe_load(incx)),
                     y, Int(unsafe_load(incy)))
end

# Mixed-precision real dot.
Base.@ccallable function dsdot_(n::Ptr{BlasInt},
        x::Ptr{Float32}, incx::Ptr{BlasInt},
        y::Ptr{Float32}, incy::Ptr{BlasInt})::Float64
    return _dot_impl(Float64, Int(unsafe_load(n)),
                     x, Int(unsafe_load(incx)),
                     y, Int(unsafe_load(incy)))
end

Base.@ccallable function sdsdot_(n::Ptr{BlasInt}, sb::Ptr{Float32},
        x::Ptr{Float32}, incx::Ptr{BlasInt},
        y::Ptr{Float32}, incy::Ptr{BlasInt})::Float32
    return _sdsdot_impl(Int(unsafe_load(n)), unsafe_load(sb),
                        x, Int(unsafe_load(incx)),
                        y, Int(unsafe_load(incy)))
end

# Complex dot — Fortran convention returns via a hidden out-pointer first
# arg (gfortran / f2c). Both unconjugated (?dotu) and conjugated (?dotc).
for (suffix, T) in ((:c, ComplexF32), (:z, ComplexF64))
    sym_u = Symbol(suffix, "dotu_")
    sym_c = Symbol(suffix, "dotc_")
    @eval begin
        Base.@ccallable function $sym_u(result::Ptr{$T}, n::Ptr{BlasInt},
                x::Ptr{$T}, incx::Ptr{BlasInt},
                y::Ptr{$T}, incy::Ptr{BlasInt})::Cvoid
            r = _dot_impl($T, Int(unsafe_load(n)),
                          x, Int(unsafe_load(incx)),
                          y, Int(unsafe_load(incy)))
            unsafe_store!(result, r)
            return
        end

        Base.@ccallable function $sym_c(result::Ptr{$T}, n::Ptr{BlasInt},
                x::Ptr{$T}, incx::Ptr{BlasInt},
                y::Ptr{$T}, incy::Ptr{BlasInt})::Cvoid
            r = _dotc_impl(Int(unsafe_load(n)),
                           x, Int(unsafe_load(incx)),
                           y, Int(unsafe_load(incy)))
            unsafe_store!(result, r)
            return
        end
    end
end

# nrm2 — real returns even for complex inputs.
Base.@ccallable function snrm2_(n::Ptr{BlasInt},
        x::Ptr{Float32}, incx::Ptr{BlasInt})::Float32
    return _nrm2_real_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

Base.@ccallable function dnrm2_(n::Ptr{BlasInt},
        x::Ptr{Float64}, incx::Ptr{BlasInt})::Float64
    return _nrm2_real_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

Base.@ccallable function scnrm2_(n::Ptr{BlasInt},
        x::Ptr{ComplexF32}, incx::Ptr{BlasInt})::Float32
    return _nrm2_complex_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

Base.@ccallable function dznrm2_(n::Ptr{BlasInt},
        x::Ptr{ComplexF64}, incx::Ptr{BlasInt})::Float64
    return _nrm2_complex_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

# asum — real returns; complex variant uses |Re|+|Im| per reference BLAS.
Base.@ccallable function sasum_(n::Ptr{BlasInt},
        x::Ptr{Float32}, incx::Ptr{BlasInt})::Float32
    return _asum_real_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

Base.@ccallable function dasum_(n::Ptr{BlasInt},
        x::Ptr{Float64}, incx::Ptr{BlasInt})::Float64
    return _asum_real_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

Base.@ccallable function scasum_(n::Ptr{BlasInt},
        x::Ptr{ComplexF32}, incx::Ptr{BlasInt})::Float32
    return _asum_complex_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

Base.@ccallable function dzasum_(n::Ptr{BlasInt},
        x::Ptr{ComplexF64}, incx::Ptr{BlasInt})::Float64
    return _asum_complex_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

# i?amax — 1-indexed argmax; 0 on invalid args.
Base.@ccallable function isamax_(n::Ptr{BlasInt},
        x::Ptr{Float32}, incx::Ptr{BlasInt})::BlasInt
    return _iamax_real_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

Base.@ccallable function idamax_(n::Ptr{BlasInt},
        x::Ptr{Float64}, incx::Ptr{BlasInt})::BlasInt
    return _iamax_real_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

Base.@ccallable function icamax_(n::Ptr{BlasInt},
        x::Ptr{ComplexF32}, incx::Ptr{BlasInt})::BlasInt
    return _iamax_complex_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

Base.@ccallable function izamax_(n::Ptr{BlasInt},
        x::Ptr{ComplexF64}, incx::Ptr{BlasInt})::BlasInt
    return _iamax_complex_impl(Int(unsafe_load(n)), x, Int(unsafe_load(incx)))
end

# ─── BLAS Level 1 — Givens family ─────────────────────────────────────────

# rot: apply Givens. srot/drot are real; csrot/zdrot apply real (c, s) to
# complex vectors — same arithmetic, types differ.
for (sym, Tv, Tc) in ((:srot_,  Float32,    Float32),
                       (:drot_,  Float64,    Float64),
                       (:csrot_, ComplexF32, Float32),
                       (:zdrot_, ComplexF64, Float64))
    @eval Base.@ccallable function $sym(n::Ptr{BlasInt},
            x::Ptr{$Tv}, incx::Ptr{BlasInt},
            y::Ptr{$Tv}, incy::Ptr{BlasInt},
            c::Ptr{$Tc}, s::Ptr{$Tc})::Cvoid
        _rot_impl!(Int(unsafe_load(n)),
                   x, Int(unsafe_load(incx)),
                   y, Int(unsafe_load(incy)),
                   unsafe_load(c), unsafe_load(s))
        return
    end
end

# rotg (real)
for (sym, T) in ((:srotg_, Float32), (:drotg_, Float64))
    @eval Base.@ccallable function $sym(a::Ptr{$T}, b::Ptr{$T},
                                          c::Ptr{$T}, s::Ptr{$T})::Cvoid
        _rotg_impl!(a, b, c, s)
        return
    end
end

# rotg (complex): c is real, s is complex
for (sym, T, TR) in ((:crotg_, ComplexF32, Float32),
                      (:zrotg_, ComplexF64, Float64))
    @eval Base.@ccallable function $sym(a::Ptr{$T}, b::Ptr{$T},
                                          c::Ptr{$TR}, s::Ptr{$T})::Cvoid
        _rotg_complex_impl!(a, b, c, s)
        return
    end
end

# rotm: apply modified Givens (real only)
for (sym, T) in ((:srotm_, Float32), (:drotm_, Float64))
    @eval Base.@ccallable function $sym(n::Ptr{BlasInt},
            x::Ptr{$T}, incx::Ptr{BlasInt},
            y::Ptr{$T}, incy::Ptr{BlasInt},
            param::Ptr{$T})::Cvoid
        _rotm_impl!(Int(unsafe_load(n)),
                    x, Int(unsafe_load(incx)),
                    y, Int(unsafe_load(incy)),
                    param)
        return
    end
end

# rotmg: generate modified Givens (real only)
for (sym, T) in ((:srotmg_, Float32), (:drotmg_, Float64))
    @eval Base.@ccallable function $sym(d1::Ptr{$T}, d2::Ptr{$T},
                                          x1::Ptr{$T}, y1::Ptr{$T},
                                          param::Ptr{$T})::Cvoid
        _rotmg_impl!(d1, d2, x1, y1, param)
        return
    end
end

# ─── Level 2 BLAS — general matrix-vector core ────────────────────────────

# gemv: y := α·op(A)·x + β·y
# args: (trans, m, n, alpha, A, lda, x, incx, beta, y, incy) + 1 hidden trans len
for (suffix, T) in ((:s, Float32), (:d, Float64),
                    (:c, ComplexF32), (:z, ComplexF64))
    sym = Symbol(suffix, "gemv_")
    @eval Base.@ccallable function $sym(trans::Ptr{UInt8},
            m::Ptr{BlasInt}, n::Ptr{BlasInt},
            α::Ptr{$T}, A::Ptr{$T}, lda::Ptr{BlasInt},
            x::Ptr{$T}, incx::Ptr{BlasInt},
            β::Ptr{$T}, y::Ptr{$T}, incy::Ptr{BlasInt},
            _lt::Csize_t)::Cvoid
        _gemv_impl!(unsafe_load(trans),
                    Int(unsafe_load(m)), Int(unsafe_load(n)),
                    unsafe_load(α), A, Int(unsafe_load(lda)),
                    x, Int(unsafe_load(incx)),
                    unsafe_load(β), y, Int(unsafe_load(incy)))
        return
    end
end

# ger / geru: A := α·x·yᵀ + A   (no conjugation)
# args: (m, n, alpha, x, incx, y, incy, A, lda) — no char args, no hidden lens
for (sym, T) in ((:sger_,  Float32),
                  (:dger_,  Float64),
                  (:cgeru_, ComplexF32),
                  (:zgeru_, ComplexF64))
    @eval Base.@ccallable function $sym(m::Ptr{BlasInt}, n::Ptr{BlasInt},
            α::Ptr{$T}, x::Ptr{$T}, incx::Ptr{BlasInt},
            y::Ptr{$T}, incy::Ptr{BlasInt},
            A::Ptr{$T}, lda::Ptr{BlasInt})::Cvoid
        _geru_impl!(Int(unsafe_load(m)), Int(unsafe_load(n)),
                    unsafe_load(α),
                    x, Int(unsafe_load(incx)),
                    y, Int(unsafe_load(incy)),
                    A, Int(unsafe_load(lda)))
        return
    end
end

# gerc: A := α·x·yᴴ + A   (conjugate y)
for (sym, T) in ((:cgerc_, ComplexF32), (:zgerc_, ComplexF64))
    @eval Base.@ccallable function $sym(m::Ptr{BlasInt}, n::Ptr{BlasInt},
            α::Ptr{$T}, x::Ptr{$T}, incx::Ptr{BlasInt},
            y::Ptr{$T}, incy::Ptr{BlasInt},
            A::Ptr{$T}, lda::Ptr{BlasInt})::Cvoid
        _gerc_impl!(Int(unsafe_load(m)), Int(unsafe_load(n)),
                    unsafe_load(α),
                    x, Int(unsafe_load(incx)),
                    y, Int(unsafe_load(incy)),
                    A, Int(unsafe_load(lda)))
        return
    end
end

# ─── Level 2 BLAS — symmetric / Hermitian ─────────────────────────────────

# symv (real): y := α·A·x + β·y, A symmetric
# args: (uplo, n, alpha, A, lda, x, incx, beta, y, incy) + 1 hidden uplo len
for (sym, T) in ((:ssymv_, Float32), (:dsymv_, Float64))
    @eval Base.@ccallable function $sym(uplo::Ptr{UInt8}, n::Ptr{BlasInt},
            α::Ptr{$T}, A::Ptr{$T}, lda::Ptr{BlasInt},
            x::Ptr{$T}, incx::Ptr{BlasInt},
            β::Ptr{$T}, y::Ptr{$T}, incy::Ptr{BlasInt},
            _lu::Csize_t)::Cvoid
        _symv_impl!(unsafe_load(uplo), Int(unsafe_load(n)),
                    unsafe_load(α), A, Int(unsafe_load(lda)),
                    x, Int(unsafe_load(incx)),
                    unsafe_load(β), y, Int(unsafe_load(incy)))
        return
    end
end

# hemv (complex): same arg layout as symv
for (sym, T) in ((:chemv_, ComplexF32), (:zhemv_, ComplexF64))
    @eval Base.@ccallable function $sym(uplo::Ptr{UInt8}, n::Ptr{BlasInt},
            α::Ptr{$T}, A::Ptr{$T}, lda::Ptr{BlasInt},
            x::Ptr{$T}, incx::Ptr{BlasInt},
            β::Ptr{$T}, y::Ptr{$T}, incy::Ptr{BlasInt},
            _lu::Csize_t)::Cvoid
        _hemv_impl!(unsafe_load(uplo), Int(unsafe_load(n)),
                    unsafe_load(α), A, Int(unsafe_load(lda)),
                    x, Int(unsafe_load(incx)),
                    unsafe_load(β), y, Int(unsafe_load(incy)))
        return
    end
end

# syr (real): A := α·x·xᵀ + A
# args: (uplo, n, alpha, x, incx, A, lda) + 1 hidden uplo len
for (sym, T) in ((:ssyr_, Float32), (:dsyr_, Float64))
    @eval Base.@ccallable function $sym(uplo::Ptr{UInt8}, n::Ptr{BlasInt},
            α::Ptr{$T}, x::Ptr{$T}, incx::Ptr{BlasInt},
            A::Ptr{$T}, lda::Ptr{BlasInt},
            _lu::Csize_t)::Cvoid
        _syr_impl!(unsafe_load(uplo), Int(unsafe_load(n)),
                   unsafe_load(α), x, Int(unsafe_load(incx)),
                   A, Int(unsafe_load(lda)))
        return
    end
end

# syr2 (real): A := α·x·yᵀ + α·y·xᵀ + A
# args: (uplo, n, alpha, x, incx, y, incy, A, lda) + 1 hidden uplo len
for (sym, T) in ((:ssyr2_, Float32), (:dsyr2_, Float64))
    @eval Base.@ccallable function $sym(uplo::Ptr{UInt8}, n::Ptr{BlasInt},
            α::Ptr{$T}, x::Ptr{$T}, incx::Ptr{BlasInt},
            y::Ptr{$T}, incy::Ptr{BlasInt},
            A::Ptr{$T}, lda::Ptr{BlasInt},
            _lu::Csize_t)::Cvoid
        _syr2_impl!(unsafe_load(uplo), Int(unsafe_load(n)),
                    unsafe_load(α),
                    x, Int(unsafe_load(incx)),
                    y, Int(unsafe_load(incy)),
                    A, Int(unsafe_load(lda)))
        return
    end
end

# her (complex Hermitian rank-1): α is REAL even when A is complex.
for (sym, T, TR) in ((:cher_, ComplexF32, Float32),
                      (:zher_, ComplexF64, Float64))
    @eval Base.@ccallable function $sym(uplo::Ptr{UInt8}, n::Ptr{BlasInt},
            α::Ptr{$TR}, x::Ptr{$T}, incx::Ptr{BlasInt},
            A::Ptr{$T}, lda::Ptr{BlasInt},
            _lu::Csize_t)::Cvoid
        _her_impl!(unsafe_load(uplo), Int(unsafe_load(n)),
                   unsafe_load(α), x, Int(unsafe_load(incx)),
                   A, Int(unsafe_load(lda)))
        return
    end
end

# her2 (complex Hermitian rank-2): α is complex.
for (sym, T) in ((:cher2_, ComplexF32), (:zher2_, ComplexF64))
    @eval Base.@ccallable function $sym(uplo::Ptr{UInt8}, n::Ptr{BlasInt},
            α::Ptr{$T}, x::Ptr{$T}, incx::Ptr{BlasInt},
            y::Ptr{$T}, incy::Ptr{BlasInt},
            A::Ptr{$T}, lda::Ptr{BlasInt},
            _lu::Csize_t)::Cvoid
        _her2_impl!(unsafe_load(uplo), Int(unsafe_load(n)),
                    unsafe_load(α),
                    x, Int(unsafe_load(incx)),
                    y, Int(unsafe_load(incy)),
                    A, Int(unsafe_load(lda)))
        return
    end
end

# ─── Level 2 BLAS — triangular ────────────────────────────────────────────

# trmv: x := op(A)·x      args: (uplo, trans, diag, n, A, lda, x, incx) + 3 hidden lens
for (suffix, T) in ((:s, Float32), (:d, Float64),
                    (:c, ComplexF32), (:z, ComplexF64))
    sym = Symbol(suffix, "trmv_")
    @eval Base.@ccallable function $sym(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
            diag::Ptr{UInt8}, n::Ptr{BlasInt},
            A::Ptr{$T}, lda::Ptr{BlasInt},
            x::Ptr{$T}, incx::Ptr{BlasInt},
            _lu::Csize_t, _lt::Csize_t, _ld::Csize_t)::Cvoid
        _trmv_impl!(unsafe_load(uplo), unsafe_load(trans), unsafe_load(diag),
                    Int(unsafe_load(n)),
                    A, Int(unsafe_load(lda)),
                    x, Int(unsafe_load(incx)))
        return
    end
end

# trsv: op(A)·x = b → x      same signature as trmv.
for (suffix, T) in ((:s, Float32), (:d, Float64),
                    (:c, ComplexF32), (:z, ComplexF64))
    sym = Symbol(suffix, "trsv_")
    @eval Base.@ccallable function $sym(uplo::Ptr{UInt8}, trans::Ptr{UInt8},
            diag::Ptr{UInt8}, n::Ptr{BlasInt},
            A::Ptr{$T}, lda::Ptr{BlasInt},
            x::Ptr{$T}, incx::Ptr{BlasInt},
            _lu::Csize_t, _lt::Csize_t, _ld::Csize_t)::Cvoid
        _trsv_impl!(unsafe_load(uplo), unsafe_load(trans), unsafe_load(diag),
                    Int(unsafe_load(n)),
                    A, Int(unsafe_load(lda)),
                    x, Int(unsafe_load(incx)))
        return
    end
end
