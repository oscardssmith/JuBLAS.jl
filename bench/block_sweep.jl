# Sweep MC and KC for the chosen kernel shape on this host.
#
# IMPORTANT: This bench uses persistent `Apack`/`Bpack` allocated once
# per (MC, KC) configuration and reused across all timed iterations. An
# earlier version allocated fresh buffers per call, which measured the
# cold-buffer regime (smaller KC wins because there's less to refetch
# from L3/RAM each call). Production code uses `gemm_workspace`-allocated
# persistent buffers, so cold-buffer numbers are misleading.
#
# We can't easily override `mc_block`/`kc_block` from outside the
# package, so the sweep inlines the `gemm!` loop with caller-supplied
# MC, KC, NC.
#
# Usage:
#     julia --project=bench bench/block_sweep.jl [N] [ITERS] [ELT]
# where ELT ∈ {Float32, Float64, ComplexF32, ComplexF64}.

using JuBLAS, LinearAlgebra, Printf, Random

const N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
const ITERS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
const ELT   = length(ARGS) >= 3 ? Symbol(ARGS[3])     : :Float64

const T = ELT === :Float32    ? Float32    :
          ELT === :Float64    ? Float64    :
          ELT === :ComplexF32 ? ComplexF32 :
          ELT === :ComplexF64 ? ComplexF64 :
          error("unknown eltype: $ELT  (expected Float32, Float64, ComplexF32, ComplexF64)")

# Real packs into `Vector{T}`; complex packs Re/Im sub-panels into
# `Vector{TR}` of double the element count.
const TR            = T <: Complex ? real(T) : T
const PACK_FACTOR   = T <: Complex ? 2 : 1
const FLOPS_PER_MAC = T <: Complex ? 8 : 2

BLAS.set_num_threads(1)
Random.seed!(0xc0ffee)

const A = randn(T, N, N)
const B = randn(T, N, N)
const C = zeros(T, N, N)

const KERNEL = JuBLAS.default_kernel(T)
@info "kernel: $(typeof(KERNEL))"

const MR_ = JuBLAS.mr(KERNEL)
const NR_ = JuBLAS.nr(KERNEL)

# Run one `gemm!` call's worth of work with caller-supplied (MC, KC, NC),
# using *pre-allocated* `Apack`/`Bpack`. Sized for the bench's largest
# grid point so the same buffers serve every config.
function run_gemm!(C, A, B, kernel, MC, KC, NC, αT, Apack, Bpack)
    M, N = size(C)
    K = size(A, 2)
    MR = JuBLAS.mr(kernel); NR = JuBLAS.nr(kernel)
    JuBLAS._scale!(C, false)

    @inbounds for jc in 1:NC:N
        nc = min(NC, N - jc + 1)
        for pc in 1:KC:K
            kc = min(KC, K - pc + 1)
            JuBLAS._pack_B!(Bpack, B, pc, jc, kc, nc, Val(NR))
            for ic in 1:MC:M
                mc = min(MC, M - ic + 1)
                JuBLAS._pack_A!(Apack, A, ic, pc, mc, kc, Val(MR))
                JuBLAS._macrokernel!(kernel, C, Apack, Bpack, ic, jc, mc, nc, kc, αT)
            end
        end
    end
    return C
end

# Default block sizes (for reference + buffer sizing).
const caches = JuBLAS._cache_sizes()
const KC_DEFAULT = JuBLAS.kc_block(KERNEL, caches)
const MC_DEFAULT = JuBLAS.mc_block(KERNEL, caches)
const NC_DEFAULT = JuBLAS.nc_block(KERNEL, caches)

# Sweep grid.
const MC_GRID = unique(sort([
    MC_DEFAULT,
    MR_, MR_ * 2, MR_ * 3, MR_ * 4, MR_ * 6, MR_ * 8, MR_ * 10, MR_ * 12,
    MR_ * 14, MR_ * 16, MR_ * 20, MR_ * 24, MR_ * 32,
]))

const KC_GRID = unique(sort([
    KC_DEFAULT,
    128, 192, 256, 320, 384, 512,
]))

const MC_MAX = maximum(MC_GRID)
const KC_MAX = maximum(KC_GRID)
const NC_FOR_BENCH = NC_DEFAULT
const APACK = Vector{TR}(undef, PACK_FACTOR * cld(MC_MAX, MR_) * MR_ * KC_MAX)
const BPACK = Vector{TR}(undef, PACK_FACTOR * cld(NC_FOR_BENCH, NR_) * NR_ * KC_MAX)

function bench_blocks(MC, KC, NC)
    αT = convert(T, true)
    for _ in 1:5
        run_gemm!(C, A, B, KERNEL, MC, KC, NC, αT, APACK, BPACK)
    end
    GC.gc()
    t0 = time_ns()
    for _ in 1:ITERS
        run_gemm!(C, A, B, KERNEL, MC, KC, NC, αT, APACK, BPACK)
    end
    elapsed = (time_ns() - t0) / 1e9
    gflops = FLOPS_PER_MAC * N^3 * ITERS / elapsed / 1e9
    return gflops
end

@printf("\n=== Block-size sweep (T=%s, N=%d, iters=%d) ===\n", T, N, ITERS)
@printf("kernel=%s  defaults: MC=%d, KC=%d, NC=%d\n",
        typeof(KERNEL), MC_DEFAULT, KC_DEFAULT, NC_DEFAULT)
@printf("\n%6s  %6s  %8s  %8s\n", "MC", "KC", "GFLOPS", "delta%")

reference = bench_blocks(MC_DEFAULT, KC_DEFAULT, NC_FOR_BENCH)
@printf("%6d  %6d  %8.1f  %8s  (default)\n", MC_DEFAULT, KC_DEFAULT, reference, "—")

for KC in KC_GRID
    for MC in MC_GRID
        (KC == KC_DEFAULT && MC == MC_DEFAULT) && continue
        gflops = bench_blocks(MC, KC, NC_FOR_BENCH)
        delta = (gflops - reference) / reference * 100
        @printf("%6d  %6d  %8.1f  %+7.1f%%\n", MC, KC, gflops, delta)
    end
end
