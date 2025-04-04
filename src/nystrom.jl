import KernelFunctions
import LinearAlgebra: inv, Diagonal, diagind
import Random

# ==========================================
# Nyström Approximation
# ==========================================

"""
    NystromApproximation{T<:Real}

A structure representing the Nyström approximation of a kernel matrix.

The approximation allows efficient matrix-vector products with an approximate kernel matrix,
using the formula K ≈ K_nm * K_mm_inv * K_nm^T.

# Data Format
- This implementation expects data with features in rows and observations in columns.
- Matrix X should be size (n_features × n_samples).

# Fields
- `landmarks::Matrix{T}`: Landmark points (n_features × m)
- `K_nm::Matrix{T}`: Kernel matrix between data and landmarks (n × m)
- `K_mm_inv::Matrix{T}`: Inverse of kernel matrix between landmarks (m × m)
"""
struct NystromApproximation{T<:Real}
    landmarks::Matrix{T}
    K_nm::Matrix{T}
    K_mm_inv::Matrix{T}

    function NystromApproximation(landmarks::Matrix{T}, 
                                K_nm::Matrix{T}, 
                                K_mm_inv::Matrix{T}) where {T<:Real}
        new{T}(landmarks, K_nm, K_mm_inv)
    end
end

"""
    multiply(approx::NystromApproximation, v::AbstractVector{<:Real})

Approximate the product of the kernel matrix with a vector using the Nyström approximation.

# Arguments
- `approx::NystromApproximation`: The Nyström approximation
- `v::AbstractVector{<:Real}`: Vector to multiply with the kernel matrix

# Returns
- `::Vector`: The approximate matrix-vector product K*v

# Throws
- `DimensionMismatch`: If dimensions of v don't match K_nm
"""
function multiply(approx::NystromApproximation{T}, 
                v::AbstractVector{<:Real}) where {T<:Real}
    n = size(approx.K_nm, 1)
    length(v) == n || throw(DimensionMismatch(
        "Vector length ($(length(v))) must match number of rows in K_nm ($n)"))
    
    # K * v ≈ K_nm * K_mm^(-1) * K_nm^T * v
    return approx.K_nm * (approx.K_mm_inv * (approx.K_nm' * v))
end

"""
    multiply_with_projection(approx::NystromApproximation, v::AbstractVector{<:Real}, W::AbstractMatrix{<:Real})

Multiply the kernel matrix with a vector after applying a projection operator.
Computes (I - W(W^T W)^(-1)W^T) * K * v using the Nyström approximation.

# Arguments
- `approx::NystromApproximation`: The Nyström approximation
- `v::AbstractVector{<:Real}`: Vector to multiply with the kernel matrix
- `W::AbstractMatrix{<:Real}`: Projection matrix

# Returns
- `::Vector`: The projected matrix-vector product
"""
function multiply_with_projection(approx::NystromApproximation{T}, 
                                v::AbstractVector{<:Real},
                                W::AbstractMatrix{<:Real}) where {T<:Real}
    # Calculate W(W^T W)^(-1)W^T * v
    WtW = W' * W
    Wv = W' * v
    WtW_inv_Wv = WtW \ Wv
    W_WtW_inv_Wv = W * WtW_inv_Wv

    # Return (I - W(W^T W)^(-1)W^T) * K * v
    return multiply(approx, v - W_WtW_inv_Wv)
end

"""
    multiply_validation(approx::NystromApproximation, X_val::AbstractMatrix{<:Real}, v::AbstractVector{<:Real}, kernel::KernelFunctions.Kernel)

Apply the Nyström approximation to compute the matrix-vector product between
a validation set and the training set.

# Arguments
- `approx::NystromApproximation`: The Nyström approximation
- `X_val::AbstractMatrix{<:Real}`: Validation data points (n_features × n_val)
- `v::AbstractVector{<:Real}`: Vector to multiply with (length n_train)
- `kernel::KernelFunctions.Kernel`: Kernel function

# Returns
- `::Vector`: The approximate matrix-vector product K_val_train * v

# Throws
- `DimensionMismatch`: If input dimensions are incompatible
"""
function multiply_validation(approx::NystromApproximation{T}, 
                          X_val::AbstractMatrix{<:Real},
                          v::AbstractVector{<:Real}, 
                          kernel::KernelFunctions.Kernel) where {T<:Real}
    # Validate input dimensions
    size(X_val, 1) == size(approx.landmarks, 1) || throw(DimensionMismatch(
        "Validation data must have the same number of features as landmarks"))
    length(v) == size(approx.K_nm, 1) || throw(DimensionMismatch(
        "Vector length must match the number of training points"))

    # Compute K_val_m: kernel matrix between validation points and landmarks
    K_val_m = KernelFunctions.kernelmatrix(kernel, X_val, approx.landmarks, obsdim=2)

    # Approximate K_val_train * v = K_val_m * K_mm^(-1) * K_nm^T * v
    return K_val_m * (approx.K_mm_inv * (approx.K_nm' * v))
end

"""
    compute_nystrom_landmarks(X::AbstractMatrix{<:Real}, num_landmarks::Int; kwargs...)

Compute landmark points for the Nyström approximation using mini-batch k-means clustering.

# Arguments
- `X::AbstractMatrix{<:Real}`: Data matrix (n_features × n_samples)
- `num_landmarks::Int`: Number of landmark points to select

# Keyword Arguments
- `batch_size::Int=100`: Mini-batch size for k-means
- `num_iterations::Int=100`: Number of k-means iterations
- `seed::Union{Int,Nothing}=nothing`: Random seed for reproducibility (optional)

# Returns
- `::Matrix`: Landmark points (n_features × m)

# Throws
- `ArgumentError`: If num_landmarks is not positive
"""
function compute_nystrom_landmarks(X::AbstractMatrix{T}, 
                                num_landmarks::Int;
                                batch_size::Int=100,
                                num_iterations::Int=100,
                                seed::Union{Int,Nothing}=nothing) where {T<:Real}
    # Validate inputs
    num_landmarks > 0 || throw(ArgumentError("Number of landmarks must be positive"))
    batch_size > 0 || throw(ArgumentError("Batch size must be positive"))
    num_iterations > 0 || throw(ArgumentError("Number of iterations must be positive"))
    
    n_features, n_samples = size(X)
    batch_size = min(batch_size, n_samples)

    if num_landmarks > n_samples
        @warn "Number of landmarks exceeds number of data points, using all data points as landmarks"
        return X
    end

    # Create RNG with seed if provided
    if seed !== nothing
        rng = Random.MersenneTwister(seed)
    else
        rng = Random.GLOBAL_RNG
    end

    # Use mini-batch k-means to find landmark points
    # Note: For our k-means implementation, we need to pass observations in columns
    # which matches our current format (n_features × n_samples)
    result = mini_batch_kmeans(X, num_landmarks, 
                             batch_size=batch_size, 
                             max_iterations=num_iterations,
                             seed=seed)

    # Check for success
    if !result.success
        @warn "K-means clustering failed. Using random sampling for landmarks."
        indices = Random.shuffle(rng, 1:n_samples)[1:num_landmarks]
        return X[:, indices]
    end

    # Return centers (already in the right format: n_features × num_landmarks)
    return result.centers
end

"""
    compute_nystrom_approximation(X::AbstractMatrix{<:Real}, landmarks::AbstractMatrix{<:Real}, kernel::KernelFunctions.Kernel; kwargs...)

Compute the Nyström approximation for a kernel matrix.

# Arguments
- `X::AbstractMatrix{<:Real}`: Data matrix (n_features × n_samples)
- `landmarks::AbstractMatrix{<:Real}`: Landmark points (n_features × m)
- `kernel::KernelFunctions.Kernel`: Kernel function

# Keyword Arguments
- `regularization::Real=1e-6`: Regularization parameter for numerical stability

# Returns
- `::NystromApproximation`: The Nyström approximation

# Throws
- `ArgumentError`: If regularization is negative
- `DimensionMismatch`: If dimensions of X and landmarks are incompatible
"""
function compute_nystrom_approximation(X::AbstractMatrix{<:Real}, 
                                    landmarks::AbstractMatrix,
                                    kernel::KernelFunctions.Kernel;
                                    regularization::Real=1e-6)
    # Get and validate types
    TX = eltype(X)
    TL = eltype(landmarks)
    T = promote_type(TX, TL)
    
    # Validate inputs
    regularization >= zero(T) || throw(ArgumentError("Regularization parameter must be non-negative"))
    
    # Convert landmarks to standard matrix format if needed
    if landmarks isa Transpose
        landmarks_matrix = Matrix{T}(landmarks)
    else
        landmarks_matrix = convert(Matrix{T}, landmarks)
    end
    
    # Ensure feature dimensions match
    size(X, 1) == size(landmarks_matrix, 1) || throw(DimensionMismatch(
        "Data points and landmarks must have the same number of features"))

    # Compute kernel matrix between data and landmarks: K_nm
    K_nm = KernelFunctions.kernelmatrix(kernel, X, landmarks_matrix, obsdim=2)

    # Compute kernel matrix between landmarks: K_mm
    K_mm = KernelFunctions.kernelmatrix(kernel, landmarks_matrix, landmarks_matrix, obsdim=2)

    # Add regularization to diagonal for numerical stability
    K_mm[diagind(K_mm)] .+= convert(T, regularization)

    # Compute inverse of K_mm (with robust fallback)
    local K_mm_inv
    try
        K_mm_inv = inv(K_mm)
    catch e
        @warn "Matrix inversion failed in Nyström approximation, using pseudo-inverse (SVD)" exception=e
        # Compute pseudo-inverse using SVD
        F = svd(K_mm)
        s = F.S
        # Set threshold for numerical stability
        epsilon = eps(T) * max(size(K_mm)...) * maximum(s)
        s_inv = [x > epsilon ? one(T)/x : zero(T) for x in s]
        K_mm_inv = F.V * Diagonal(s_inv) * F.U'
    end

    return NystromApproximation(landmarks_matrix, K_nm, K_mm_inv)
end