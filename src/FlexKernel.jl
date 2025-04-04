module FlexKernel

using LinearAlgebra
using Distances
using Random
using StatsBase
using KernelFunctions
using CSV
using DataFrames

"""
FlexKernel.jl

A Julia package for kernel methods with efficient approximations.

This package includes:
- Custom kernel implementations (SincKernel)
- Nyström approximation for large-scale kernel matrices
- Mini-batch k-means clustering

Data Format:
- All functions expect data with features in rows and observations in columns
- Matrices should have shape (n_features × n_samples)

Example:
```julia
using FlexKernel

# Create a kernel
k = SincKernel(1.0)

# Generate data (features in rows, samples in columns)
X = randn(2, 100)  # 2 features, 100 samples
Y = randn(2, 50)   # 2 features, 50 samples

# Compute kernel matrix
K = kernelmatrix(k, X, Y, obsdim=2)

# Compute Nyström approximation
landmarks = compute_nystrom_landmarks(X, 10)
nystrom = compute_nystrom_approximation(X, landmarks, k)

# Use approximation
v = randn(100)
Kv = multiply(nystrom, v)
```
"""
FlexKernel

# Include component files
include("kernels.jl")
include("nystrom.jl")
include("kmeans.jl")
include("utilities.jl")

# Re-export KernelFunctions for user convenience
export KernelFunctions

# Export custom kernel and helper functions
export SincKernel, sinc_value

# Export Nyström approximation
export NystromApproximation, compute_nystrom_landmarks, compute_nystrom_approximation,
       multiply, multiply_with_projection, multiply_validation

# Export K-means
export mini_batch_kmeans, KMeansResult

# Export utilities
export prepare_matrix, to_dataframe

end # module