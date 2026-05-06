module JuBLAS

using CpuId
using SIMD

include("utils.jl")
include("gemm.jl")

export gemm!, gemm_workspace, default_kernel,
       AbstractKernel, ScalarKernel, SIMDKernel
end
