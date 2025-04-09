using Plots
using LinearAlgebra
using Distances

# Internal function for assigning to nearest landmark
function _assign_to_nearest(X::AbstractMatrix, landmarks::AbstractMatrix, distance::Distances.SemiMetric=Distances.Euclidean())
    n_samples = size(X, 2)
    m = size(landmarks, 2)
    assignments = zeros(Int, n_samples)
    for i in 1:n_samples
        min_dist = Inf
        best_cluster = 1
        for j in 1:m
            dist = Distances.evaluate(distance, X[:,i], landmarks[:,j])
            if dist < min_dist
                min_dist = dist
                best_cluster = j
            end
        end
        assignments[i] = best_cluster
    end
    return assignments
end

"""
    plot_kernel_1d(kernel, x0::Real, x_range::AbstractVector{<:Real}; kwargs...)

Plot the kernel function k(x0, x) for x in x_range.

# Arguments
- `kernel`: The kernel function or object from KernelFunctions.jl or FlexKernel.
- `x0::Real`: The fixed point (scalar).
- `x_range::AbstractVector{<:Real}`: The range of x values to plot.

# Keyword Arguments
- `kwargs...`: Additional arguments passed to `plot` (e.g., `title`, `xlabel`, `ylabel`, `label`).

# Examples
```julia
using FlexKernel, KernelFunctions, Plots
k = SqExponentialKernel()
plot_kernel_1d(k, 0.0, -5:0.1:5, title="RBF Kernel", label="RBF")
```
"""
function plot_kernel_1d(kernel, x0::Real, x_range::AbstractVector{<:Real}; kwargs...)
    values = [kernel([x0], [x]) for x in x_range]
    plot(x_range, values; kwargs...)
end

"""
    compare_kernels_1d(kernels, x0::Real, x_range::AbstractVector{<:Real}; labels=nothing, kwargs...)

Plot multiple kernel functions k(x0, x) for x in x_range on the same figure.

# Arguments
- `kernels`: A list of kernel functions or objects.
- `x0::Real`: The fixed point (scalar).
- `x_range::AbstractVector{<:Real}`: The range of x values to plot.

# Keyword Arguments
- `labels`: Optional list of labels for each kernel (defaults to "Kernel 1", "Kernel 2", ...).
- `kwargs...`: Additional arguments passed to `plot` (e.g., `title`, `xlabel`, `ylabel`).

# Examples
```julia
using FlexKernel, KernelFunctions, Plots
k1 = SqExponentialKernel()
k2 = SincKernel(1.0)
compare_kernels_1d([k1, k2], 0.0, -5:0.1:5, labels=["RBF", "Sinc"], title="Kernel Comparison")
```
"""
function compare_kernels_1d(kernels, x0::Real, x_range::AbstractVector{<:Real}; labels=nothing, kwargs...)
    p = plot()
    for (i, k) in enumerate(kernels)
        values = [k([x0], [x]) for x in x_range]
        label = labels === nothing ? "Kernel $i" : labels[i]
        plot!(p, x_range, values, label=label; kwargs...)
    end
    return p
end

"""
    plot_kernel_2d(kernel, x0::AbstractVector{<:Real}, x1_range::AbstractVector{<:Real}, x2_range::AbstractVector{<:Real}; kwargs...)

Plot a heatmap of the kernel function k(x0, x) for x in a 2D grid defined by x1_range and x2_range.

# Arguments
- `kernel`: The kernel function or object.
- `x0::AbstractVector{<:Real}`: The fixed point (2D vector).
- `x1_range::AbstractVector{<:Real}`: The range for the first dimension.
- `x2_range::AbstractVector{<:Real}`: The range for the second dimension.

# Keyword Arguments
- `kwargs...`: Additional arguments passed to `heatmap` (e.g., `title`, `xlabel`, `ylabel`).

# Examples
```julia
using FlexKernel, KernelFunctions, Plots
k = SqExponentialKernel()
plot_kernel_2d(k, [0.0, 0.0], -5:0.1:5, -5:0.1:5, title="RBF Kernel Heatmap")
```
"""
function plot_kernel_2d(kernel, x0::AbstractVector{<:Real}, x1_range::AbstractVector{<:Real}, x2_range::AbstractVector{<:Real}; kwargs...)
    @assert length(x0) == 2 "x0 must be a 2D vector"
    values = [kernel(x0, [x1, x2]) for x1 in x1_range, x2 in x2_range]
    heatmap(x1_range, x2_range, values; kwargs...)
end

"""
    plot_landmarks_2d(X::AbstractMatrix, landmarks::AbstractMatrix; color_by_cluster=true, distance::Distances.SemiMetric=Distances.Euclidean(), kwargs...)

Plot the data points and landmarks in 2D.

# Arguments
- `X::AbstractMatrix`: Data matrix (2 × n_samples).
- `landmarks::AbstractMatrix`: Landmark points (2 × m).

# Keyword Arguments
- `color_by_cluster::Bool=true`: If true, color data points by their assigned cluster based on nearest landmark.
- `distance::Distances.SemiMetric=Distances.Euclidean()`: Distance metric for cluster assignment.
- `kwargs...`: Additional arguments passed to `scatter` (e.g., `title`, `xlabel`, `ylabel`).

# Examples
```julia
using FlexKernel, Plots
X = randn(2, 100)
landmarks = compute_nystrom_landmarks(X, 5)
plot_landmarks_2d(X, landmarks, title="Landmark Distribution")
```
"""
function plot_landmarks_2d(X::AbstractMatrix, landmarks::AbstractMatrix; color_by_cluster=true, distance::Distances.SemiMetric=Distances.Euclidean(), kwargs...)
    @assert size(X,1) == 2 "Data must be 2D"
    @assert size(landmarks,1) == 2 "Landmarks must be 2D"
    p = plot()
    if color_by_cluster
        assignments = _assign_to_nearest(X, landmarks, distance)
        unique_assignments = unique(assignments)
        for assign in unique_assignments
            idx = findall(assignments .== assign)
            scatter!(p, X[1,idx], X[2,idx], label="Cluster $assign", markersize=3)
        end
    else
        scatter!(p, X[1,:], X[2,:], label="Data", markersize=3)
    end
    scatter!(p, landmarks[1,:], landmarks[2,:], label="Landmarks", marker=:star, markersize=8, color=:red)
    return p
end

"""
    plot_nystrom_accuracy(nystrom::NystromApproximation, X::AbstractMatrix, kernel; compute_full=false, max_n_full=1000, kwargs...)

Plot the eigenvalues of the full kernel matrix and the Nystrom approximation for diagnostic purposes.

# Arguments
- `nystrom::NystromApproximation`: The Nystrom approximation object.
- `X::AbstractMatrix`: Data matrix (n_features × n_samples).
- `kernel`: The kernel function or object.

# Keyword Arguments
- `compute_full::Bool=false`: If true, compute eigenvalues of the full kernel matrix.
- `max_n_full::Int=1000`: Maximum number of samples for computing full eigenvalues (if `compute_full` is false, computes only if n ≤ max_n_full).
- `kwargs...`: Additional arguments passed to `plot` (e.g., `title`, `xlabel`, `ylabel`).

# Examples
```julia
using FlexKernel, KernelFunctions, Plots
X = randn(2, 50)
kernel = SqExponentialKernel()
landmarks = compute_nystrom_landmarks(X, 10)
nystrom = compute_nystrom_approximation(X, landmarks, kernel)
plot_nystrom_accuracy(nystrom, X, kernel, compute_full=true, title="Nystrom Approximation Accuracy")
```
"""
function plot_nystrom_accuracy(nystrom::NystromApproximation, X::AbstractMatrix, kernel; compute_full=false, max_n_full=1000, kwargs...)
    n = size(X,2)
    if compute_full || n <= max_n_full
        K_full = kernelmatrix(kernel, X, X, obsdim=2)
        eigvals_full = eigvals(Symmetric(K_full))
        sort!(eigvals_full, rev=true)
    else
        eigvals_full = nothing
    end
    
    # Compute approximated eigenvalues
    K_mm_inv = nystrom.K_mm_inv
    K_nm = nystrom.K_nm
    landmarks = nystrom.landmarks
    K_mm = kernelmatrix(kernel, landmarks, landmarks, obsdim=2)
    regularization = 1e-6  # Matches default in compute_nystrom_approximation
    K_mm[diagind(K_mm)] .+= regularization
    eigvals_Kmm, V = eigen(Symmetric(K_mm))
    eigvals_Kmm = max.(eigvals_Kmm, 0)  # Handle numerical issues
    K_mm_inv_sqrt = V * Diagonal(1 ./ sqrt.(eigvals_Kmm .+ 1e-10)) * V'
    M = K_mm_inv_sqrt * (K_nm' * K_nm) * K_mm_inv_sqrt
    eigvals_approx = eigvals(Symmetric(M))
    sort!(eigvals_approx, rev=true)
    
    # Plot
    p = plot()
    if eigvals_full !== nothing
        k = min(length(eigvals_full), length(eigvals_approx))
        plot!(p, 1:k, eigvals_full[1:k], label="Full K", marker=:circle)
    end
    plot!(p, 1:length(eigvals_approx), eigvals_approx, label="Nystrom approx", marker=:square)
    xlabel!(p, "Eigenvalue index")
    ylabel!(p, "Eigenvalue")
    return p
end