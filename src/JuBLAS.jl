module JuBLAS

include("gemm.jl")

export gemm!, default_kernel,
       AbstractKernel, ScalarKernel, SIMDKernel
end
