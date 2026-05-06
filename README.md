# JuBLAS

A pure-Julia BLAS implementation. Goto/BLIS-style cache-blocked microkernels
written with `SIMD.jl` and `@generated` unrolling.

## Status

Single-threaded `gemm!` for real and complex element types. The current
focus has been getting one well-tuned Level 3 routine end-to-end; the rest
of the BLAS surface is not yet built.

## Usage

```julia
using JuBLAS

A = randn(1024, 512)
B = randn(512, 768)
C = zeros(1024, 768)

# C ← α·A·B + β·C, in place
gemm!(C, A, B, 1.0, 0.0)
```

For repeated calls, preallocate the pack buffers to keep them warm in
cache:

```julia
kernel = default_kernel(Float64)
Apack, Bpack = gemm_workspace(Float64, kernel)
for _ in 1:N
    gemm!(C, A, B, 1.0, 0.0; kernel, Apack, Bpack)
end
```

`Float64`, `Float32`, `ComplexF64`, and `ComplexF32` all dispatch to a
SIMD microkernel (AVX-512 / AVX2 / SSE2 picked at runtime). Other element
types fall through to a scalar `@generated` kernel.

## Layout

- `src/utils.jl` — kernel types, CPU/cache detection, block-size traits,
  `prefix_mask` SIMD primitives.
- `src/gemm.jl` — real `gemm!`, packing, macrokernel, scalar + SIMD
  microkernels.
- `src/gemm_complex.jl` — complex `gemm!` (struct-of-arrays Re/Im
  packing, four-FMA complex multiply).
- `bench/` — block-size and shape sweeps, OpenBLAS comparison harnesses.
- `test/runtests.jl` — correctness sweep against `LinearAlgebra.mul!`.

## TODO — unimplemented BLAS surface

### Gaps in the existing `gemm!`

- [x] Transpose / adjoint inputs (`gemm!(C, transpose(A), B)`,
      `gemm!(C, A', B)`, etc.) — `Transpose`/`Adjoint` wrappers dispatch
      to dedicated pack methods, no allocation.
- [x] `Symmetric` / `Hermitian` inputs — currently materialize the
      wrapped matrix via `Matrix(A)` and forward to `gemm!`, so they
      allocate an N×N temporary. A real `symm!`/`hemm!` kernel reading
      one triangle in place is still TODO.
- [ ] Mixed precision (`Float32 × Float32 → Float64`, etc.).
- [ ] Multi-threading.
- [ ] Non-square register tiles tuned for `M ≪ N` and `M ≫ N` regimes.
- [ ] Strided / non-`StridedMatrix` C fast path (currently routes through
      the scalar kernel).

### Level 3 (matrix-matrix)

- [ ] `symm!`  — symmetric × general
- [ ] `hemm!`  — Hermitian × general
- [ ] `syrk!`  — symmetric rank-k update (`C ← α·A·Aᵀ + β·C`)
- [ ] `herk!`  — Hermitian rank-k update
- [ ] `syr2k!` — symmetric rank-2k update
- [ ] `her2k!` — Hermitian rank-2k update
- [ ] `trmm!`  — triangular × general
- [ ] `trsm!`  — triangular solve with multiple right-hand sides

### Level 2 (matrix-vector)

- [ ] `gemv!`  — general matrix-vector multiply
- [ ] `gbmv!`  — general banded matrix-vector multiply
- [ ] `symv!` / `sbmv!` / `spmv!` — symmetric (full / banded / packed)
- [ ] `hemv!` / `hbmv!` / `hpmv!` — Hermitian (full / banded / packed)
- [ ] `trmv!` / `tbmv!` / `tpmv!` — triangular matrix-vector multiply
- [ ] `trsv!` / `tbsv!` / `tpsv!` — triangular solve
- [ ] `ger!`  — general rank-1 update
- [ ] `geru!` / `gerc!` — complex rank-1 update (unconjugated / conjugated)
- [ ] `syr!` / `spr!` — symmetric rank-1 update (full / packed)
- [ ] `her!` / `hpr!` — Hermitian rank-1 update (full / packed)
- [ ] `syr2!` / `spr2!` / `her2!` / `hpr2!` — rank-2 updates

### Level 1 (vector-vector)

- [ ] `axpy!` — `y ← α·x + y`
- [ ] `axpby!` — `y ← α·x + β·y`
- [ ] `dot` / `dotu` / `dotc` — inner product (real / complex unconj /
      complex conj)
- [ ] `nrm2` — Euclidean norm
- [ ] `asum` — sum of absolute values
- [ ] `iamax` — index of max-magnitude element
- [ ] `scal!` — `x ← α·x`
- [ ] `copy!` — BLAS-style strided copy
- [ ] `swap!` — strided swap
- [ ] `rot!` / `rotg` / `rotm!` / `rotmg` — Givens / modified Givens
      rotations

### Infrastructure

- [ ] `LinearAlgebra.BLAS` interface shims so JuBLAS routines can be
      installed as the `libblastrampoline` backend.
- [ ] Threaded variants (likely a `jc`-loop or `ic`-loop split for
      Level 3, plus a `gemv`-style row split for Level 2).
- [ ] AArch64 / NEON / SVE microkernels.
- [ ] Half-precision (`Float16`, `BFloat16`) microkernels.

## License

MIT — see `LICENSE.md`.
