using KernelFunctions
using LinearAlgebra

# ==========================================
# Custom Kernel Implementations
# ==========================================

"""
    SincKernel(bandwidth::T) where {T<:Real}

A separable kernel based on the sinc function, defined as:

```math
k(\\mathbf{x}, \\mathbf{y}) = \\prod_{i=1}^d \\text{sinc}\\left(\\frac{x_i - y_i}{\\text{bandwidth}}\\right)
```

where `sinc(z) = sin(πz)/(πz)` if `z ≠ 0` and `sinc(0) = 1`.

# Data Format
- This kernel expects data with features in rows and observations in columns.
- Use matrix X of size (n_features × n_samples).

# Arguments
- `bandwidth::T`: Positive scalar that controls the width of the kernel

# Examples
```julia
k = SincKernel(1.0)
x = [1.0, 2.0]  # Feature vector (1D)
y = [1.5, 2.5]  # Feature vector (1D)
k(x, y)  # Evaluate kernel between two points

# For matrix operations
X = rand(2, 100)  # 2 features, 100 samples
Y = rand(2, 50)   # 2 features, 50 samples
K = kernelmatrix(k, X, Y)  # Computes 100×50 kernel matrix
```
"""
struct SincKernel{T<:Real} <: KernelFunctions.SimpleKernel
    bandwidth::T
    
    function SincKernel(bandwidth::T) where {T<:Real}
        bandwidth > zero(T) || throw(ArgumentError("Bandwidth must be positive"))
        new{T}(bandwidth)
    end
end

"""
    sinc_value(x::Real)

Compute the normalized sinc function value: sin(πx)/(πx) for non-zero x, 1.0 for x ≈ 0.
Uses a numerically stable approach near zero.

# Arguments
- `x::Real`: Input value

# Returns
- `::Real`: The sinc function value
"""
function sinc_value(x::Real)
    T = typeof(x)
    if abs(x) < 10*eps(T)
        return one(T)
    else
        π_x = T(π) * x
        return sin(π_x) / π_x
    end
end

# Make sinc_value work with arrays through broadcasting
sinc_value(x::AbstractArray{T}) where {T<:Real} = sinc_value.(x)

"""
    (k::SincKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})

Evaluate the Sinc kernel between two points.

# Arguments
- `x::AbstractVector{<:Real}`: First point (feature vector)
- `y::AbstractVector{<:Real}`: Second point (feature vector)

# Returns
- `::Real`: The kernel value k(x, y)

# Throws
- `DimensionMismatch`: If x and y have different dimensions
"""
function (k::SincKernel{T})(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) where {T<:Real}
    length(x) == length(y) || throw(DimensionMismatch("Vectors must have the same dimension"))

    # Ensure type stability by getting the promoted type
    R = promote_type(T, eltype(x), eltype(y))
    result = one(R)
    
    for i in eachindex(x, y)
        diff = (x[i] - y[i]) / k.bandwidth
        result *= sinc_value(diff)
    end

    return result
end

"""
    KernelFunctions.kernelmatrix(k::SincKernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim=2)

Efficiently compute the kernel matrix between two sets of points.

# Arguments
- `k::SincKernel`: The sinc kernel
- `X::AbstractMatrix`: First set of points
- `Y::AbstractMatrix`: Second set of points
- `obsdim::Int=2`: Dimension representing observations, must be 2 (observations in columns)

# Returns
- `::Matrix`: The kernel matrix (n_samples_X × n_samples_Y)

# Throws
- `DimensionMismatch`: If X and Y have different feature dimensions
- `ArgumentError`: If obsdim is not 2 (only observations in columns is supported)
"""
function KernelFunctions.kernelmatrix(k::SincKernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim=2)
    # Only support observations in columns
    obsdim == 2 || throw(ArgumentError("SincKernel only supports obsdim=2 (observations in columns)"))
    
    d, n_x = size(X)
    d_y, n_y = size(Y)
    
    d == d_y || throw(DimensionMismatch("Feature dimensions must match"))
    
    # Ensure type stability by using promoted type
    T = promote_type(eltype(X), eltype(Y), typeof(k.bandwidth))
    K = ones(T, n_x, n_y)
    
    # Process each dimension efficiently using broadcasting
    for dim in 1:d
        # Create difference matrix using broadcasting
        diffs = (reshape(X[dim,:], :, 1) .- reshape(Y[dim,:], 1, :)) ./ k.bandwidth

        # Apply sinc function and multiply with accumulated result
        K .*= sinc_value.(diffs)
    end
    
    return K
end

"""
    KernelFunctions.kernelmatrix_diag(k::SincKernel, X::AbstractMatrix; obsdim=2)

Efficiently compute the diagonal of the kernel matrix for a set of points.

# Arguments
- `k::SincKernel`: The sinc kernel
- `X::AbstractMatrix`: Set of points
- `obsdim::Int=2`: Dimension representing observations, must be 2 (observations in columns)

# Returns
- `::Vector`: The diagonal of the kernel matrix (length n_samples)

# Throws
- `ArgumentError`: If obsdim is not 2 (only observations in columns is supported)
"""
function KernelFunctions.kernelmatrix_diag(k::SincKernel, X::AbstractMatrix; obsdim=2)
    # Only support observations in columns
    obsdim == 2 || throw(ArgumentError("SincKernel only supports obsdim=2 (observations in columns)"))
    
    _, n = size(X)
    T = promote_type(eltype(X), typeof(k.bandwidth))
    return ones(T, n)  # Sinc kernel evaluates to 1.0 when x = y
end

# These methods are required by KernelFunctions but not used by SincKernel
function KernelFunctions.kappa(::SincKernel, d2::Real)
    error("SincKernel is a separable kernel and doesn't use the standard kappa/metric approach")
end

function KernelFunctions.metric(::SincKernel)
    error("SincKernel is a separable kernel and doesn't use the standard kappa/metric approach")
end