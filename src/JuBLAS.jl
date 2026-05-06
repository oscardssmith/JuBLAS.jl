module JuBLAS

using CpuId

include("gemm.jl")

export gemm!, gemm_workspace, default_kernel,
       AbstractKernel, ScalarKernel, SIMDKernel
end
