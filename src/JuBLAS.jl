module JuBLAS

include("gemm.jl")

export gemm!, default_kernel,
       AbstractKernel, ScalarKernel, AVX512Kernel, AVX512F64Kernel
end
