using JuBLAS, LinearAlgebra, LinuxPerf, Printf, Random

const N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
const ITERS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5000
const ELT   = length(ARGS) >= 3 ? Symbol(ARGS[3])     : :Float32

BLAS.set_num_threads(1)

const T = ELT === :Float64 ? Float64 : Float32
Random.seed!(0xc0ffee)
const A = rand(T, N, N)
const B = rand(T, N, N)
const C = zeros(T, N, N)

const KERNEL = T === Float32 ? JuBLAS.SIMDKernel{16, 32, 14, Float32}() :
                                JuBLAS.SIMDKernel{8,  16, 14, Float64}()

const Apack, Bpack = JuBLAS.gemm_workspace(T, KERNEL)

const EVENTS = "(cpu-cycles,instructions,branch-instructions,branch-misses)," *
               "(L1-dcache-loads,L1-dcache-load-misses)," *
               "(LLC-loads,LLC-load-misses)"

function bench(label, work)
    GC.gc()
    t0 = time_ns()
    stats = @pstats EVENTS work()
    elapsed = (time_ns() - t0) / 1e9
    gflops  = 2 * N^3 * ITERS / elapsed / 1e9
    @printf("\n=== %s  —  %.2fs  %.1f GFLOPS  (N=%d, iters=%d, T=%s) ===\n",
            label, elapsed, gflops, N, ITERS, T)
    show(stdout, MIME"text/plain"(), stats)
    println()
end

ours_loop()    = (for _ in 1:ITERS; JuBLAS.gemm!(C, A, B; kernel=KERNEL, Apack=Apack, Bpack=Bpack); end)
mul_loop()     = (for _ in 1:ITERS; mul!(C, A, B); end)

# OpenBLAS: ships as Julia's default LBT backend, no extra import.
for _ in 1:5
    JuBLAS.gemm!(C, A, B; kernel=KERNEL, Apack=Apack, Bpack=Bpack)
    mul!(C, A, B)
end
bench("ours",     ours_loop)
bench("openblas", mul_loop)

# MKL: `using MKL` switches LBT's default loader. Subsequent `mul!` calls
# go through MKL. Loaded after the OpenBLAS bench so we get both.
@info "switching BLAS backend to MKL"
using MKL
BLAS.set_num_threads(1)
@info "active BLAS: $(BLAS.get_config())"
for _ in 1:5; mul!(C, A, B); end
bench("mkl", mul_loop)
