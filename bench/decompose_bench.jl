using JuBLAS, LinearAlgebra, Printf, Random, BenchmarkTools

# Decomposes a single `gemm!` call into its constituent phases (_scale!,
# _pack_A!, _pack_B!, _macrokernel!) and times each independently. Targets
# the small-N regime where one macrokernel call covers the whole problem,
# i.e. M, N, K each ≤ MC, KC, NC respectively. In that regime the outer
# jc/pc/ic loops in `_gemm!` execute once, so calling the internals
# directly is a faithful breakdown.
#
# `@belapsed` auto-tunes iteration counts; no manual `iters` arg.
#
# Usage:
#     julia --project=bench bench/decompose_bench.jl [N] [ELT] [W MR NR]
# If W MR NR are given, use SIMDKernel{W,MR,NR,T}; otherwise default_kernel.
#
# Caveat: phases timed in isolation see different cache state than they do
# inside the full call. If `gemm!` takes the unpacked fast path (current
# rule: K ≤ 64), the full-call time will be lower than the phase sum
# because pack-A and pack-B are entirely bypassed in the real path.

const N   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
const ELT = length(ARGS) >= 2 ? Symbol(ARGS[2])     : :Float32

const T = eval(ELT)

const FLOPS_PER_MAC = T <: Complex ? 8 : 2

BLAS.set_num_threads(1)
Random.seed!(0xc0ffee)

const KERNEL = if length(ARGS) >= 5
    W_     = parse(Int, ARGS[3])
    MR_arg = parse(Int, ARGS[4])
    NR_arg = parse(Int, ARGS[5])
    SIMDKernel{W_, MR_arg, NR_arg, T}()
else
    JuBLAS.default_kernel(T, N, N, N)
end

const Apack, Bpack = JuBLAS.gemm_workspace(T, KERNEL)
const MR_ = JuBLAS.mr(KERNEL)
const NR_ = JuBLAS.nr(KERNEL)

const A = randn(T, N, N)
const B = randn(T, N, N)
const C = zeros(T, N, N)

const M_, K_ = N, N
const αT     = one(T)

let caches = JuBLAS._cache_sizes()
    MC_ = JuBLAS.mc_block(KERNEL, caches)
    KC_ = JuBLAS.kc_block(KERNEL, caches)
    NC_ = JuBLAS.nc_block(KERNEL, caches)
    @assert N <= MC_ && N <= KC_ && N <= NC_ "N=$N exceeds one of MC=$MC_, KC=$KC_, NC=$NC_; decomposition assumption broken"
end

const BUDGET = 0.3   # seconds per phase

let Awarm = randn(T, 256, 256), Bwarm = randn(T, 256, 256), Cwarm = zeros(T, 256, 256)
    t0 = time_ns()
    while (time_ns() - t0) < 3_000_000_000
        mul!(Cwarm, Awarm, Bwarm)
    end
end

@printf("\n=== gemm! decomposition (T=%s, N=%d, kernel=%s) ===\n", T, N, KERNEL)
@printf("  MR=%d  NR=%d\n\n", MR_, NR_)

# Pre-pack so the macrokernel timing sees realistic packed inputs.
JuBLAS._pack_A!(Apack, A, 1, 1, M_, K_, Val(MR_))
JuBLAS._pack_B!(Bpack, B, 1, 1, K_, N,  Val(NR_))

t_scale = @belapsed JuBLAS._scale!($C, false) seconds=BUDGET
t_packA = @belapsed JuBLAS._pack_A!($Apack, $A, 1, 1, $M_, $K_, $(Val(MR_))) seconds=BUDGET
t_packB = @belapsed JuBLAS._pack_B!($Bpack, $B, 1, 1, $K_, $N,  $(Val(NR_))) seconds=BUDGET
t_macro = @belapsed JuBLAS._macrokernel!($KERNEL, $C, $Apack, $Bpack,
                                          1, 1, $M_, $N, $K_, $αT) seconds=BUDGET
t_full  = @belapsed JuBLAS.gemm!($C, $A, $B; kernel=$KERNEL,
                                  Apack=$Apack, Bpack=$Bpack) seconds=BUDGET
t_blas  = @belapsed mul!($C, $A, $B) seconds=BUDGET

t_sum = t_scale + t_packA + t_packB + t_macro
flops = FLOPS_PER_MAC * Float64(N)^3

@printf("%-28s  %9.1f ns/call\n", "_scale!  (β=0 → fill)", t_scale * 1e9)
@printf("%-28s  %9.1f ns/call\n", "_pack_A!",              t_packA * 1e9)
@printf("%-28s  %9.1f ns/call\n", "_pack_B!",              t_packB * 1e9)
@printf("%-28s  %9.1f ns/call\n", "_macrokernel!",         t_macro * 1e9)
@printf("%-28s  %9.1f ns/call\n", "(sum of phases)",       t_sum   * 1e9)
println()
@printf("%-28s  %9.1f ns/call\n", "gemm!  (full call)",    t_full  * 1e9)
@printf("%-28s  %9.1f ns/call\n", "openblas mul!",         t_blas  * 1e9)

@printf("\n  ours    : %7.1f GFLOPS\n", flops / t_full / 1e9)
@printf("  openblas: %7.1f GFLOPS\n",   flops / t_blas / 1e9)
@printf("  full vs phase-sum: %+6.1f%%  (%.1f vs %.1f ns)\n",
        100 * (t_full - t_sum) / t_sum, t_full * 1e9, t_sum * 1e9)

println("\n  Phase share of full gemm! call:")
for (name, t) in (("_scale!",       t_scale),
                  ("_pack_A!",      t_packA),
                  ("_pack_B!",      t_packB),
                  ("_macrokernel!", t_macro))
    @printf("    %-16s  %5.1f%%  (%6.1f ns/call)\n",
            name, 100 * t / t_full, t * 1e9)
end
