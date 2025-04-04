using Distances, Random, StatsBase

# ==========================================
# K-means Implementation
# ==========================================

"""
    KMeansResult{T<:Real}

Result of a k-means clustering procedure.

# Data Format
For column-oriented data (n_features × n_samples):
- `centers` has shape (n_features × k)
- `assignments` has length n_samples

# Fields
- `centers::Matrix{T}`: Cluster centers (n_features × k)
- `assignments::Vector{Int}`: Cluster assignment for each data point
- `success::Bool`: Whether the clustering was successful
"""
struct KMeansResult{T<:Real}
    centers::Matrix{T}
    assignments::Vector{Int}
    success::Bool
end

"""
    mini_batch_kmeans(X::AbstractMatrix{<:Real}, k::Int; kwargs...)

Perform mini-batch k-means clustering on data X.

# Data Format
- This function expects data with features in rows and observations in columns.
- Matrix X should be of size (n_features × n_samples).

# Arguments
- `X::AbstractMatrix{<:Real}`: Data matrix (n_features × n_samples)
- `k::Int`: Number of clusters

# Keyword Arguments
- `batch_size::Int=100`: Size of mini-batches
- `max_iterations::Int=100`: Maximum number of iterations
- `seed::Union{Int,Nothing}=nothing`: Random seed for reproducibility
- `rng::Random.AbstractRNG=Random.GLOBAL_RNG`: Random number generator
- `tol::Real=1e-4`: Convergence tolerance for early stopping
- `distance::Distances.SemiMetric=Distances.SqEuclidean()`: Distance metric to use

# Returns
- `::KMeansResult`: The clustering result containing centers (n_features × k)

# Throws
- `ArgumentError`: If input parameters are invalid
"""
function mini_batch_kmeans(X::AbstractMatrix{T}, k::Int;
                         batch_size::Int=100,
                         max_iterations::Int=100,
                         seed::Union{Int,Nothing}=nothing,
                         rng::Random.AbstractRNG=Random.GLOBAL_RNG,
                         tol::Real=1e-4,
                         distance::Distances.SemiMetric=Distances.SqEuclidean()) where {T<:Real}
    # Create RNG from seed if provided
    if seed !== nothing
        rng = Random.MersenneTwister(seed)
    end
    
    # Get dimensions (features in rows, samples in columns)
    n_features, n_samples = size(X)

    # Validate inputs
    k > 0 || throw(ArgumentError("Number of clusters must be positive"))
    batch_size > 0 || throw(ArgumentError("Batch size must be positive"))
    max_iterations > 0 || throw(ArgumentError("Maximum iterations must be positive"))
    batch_size = min(batch_size, n_samples)

    if n_samples < k
        @warn "Number of samples less than number of clusters"
        return KMeansResult(Matrix{T}(undef, n_features, 0), Vector{Int}(), false)
    end

    # Initialize centers using k-means++ method
    centers = kmeans_plusplus_init(X, k, distance, rng)

    # Initialize cluster counts for online updates
    counts = zeros(Int, k)
    
    # Track center changes for convergence check
    old_centers = similar(centers)
    center_shift = Inf

    # Main loop over iterations
    for iter in 1:max_iterations
        # Store old centers for convergence check
        copyto!(old_centers, centers)
        
        # Shuffle indices for batch selection
        indices = Random.shuffle(rng, 1:n_samples)

        # Process mini-batches
        for start_idx in 1:batch_size:n_samples
            end_idx = min(start_idx + batch_size - 1, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_size_actual = length(batch_indices)

            # Extract mini-batch (columns from X)
            batch = X[:, batch_indices]

            # Assign points to nearest clusters
            assignments = zeros(Int, batch_size_actual)
            for i in 1:batch_size_actual
                min_dist = Inf
                best_cluster = 1

                for j in 1:k
                    # Distance between feature vectors
                    dist = Distances.evaluate(distance, batch[:,i], centers[:,j])
                    if dist < min_dist
                        min_dist = dist
                        best_cluster = j
                    end
                end

                assignments[i] = best_cluster
            end

            # Update cluster centers using online update formula
            for i in 1:batch_size_actual
                c = assignments[i]
                eta = 1.0 / (counts[c] + 1)
                counts[c] += 1
                # Update the center using column vectors
                centers[:,c] = (1.0 - eta) * centers[:,c] + eta * batch[:,i]
            end
        end

        # Reinitialize empty clusters
        empty_clusters = findall(counts .== 0)
        for c in empty_clusters
            random_idx = rand(rng, 1:n_samples)
            centers[:,c] = X[:,random_idx]
            counts[c] = 1
        end
        
        # Check for convergence
        center_shift = 0.0
        for j in 1:k
            # Distance between old and new centers
            center_dist = Distances.evaluate(distance, centers[:,j], old_centers[:,j])
            center_shift = max(center_shift, center_dist)
        end
        
        if center_shift < tol
            @info "K-means converged after $iter iterations"
            break
        end
    end

    # Assign all points to nearest center
    assignments = zeros(Int, n_samples)
    for i in 1:n_samples
        min_dist = Inf
        best_cluster = 1

        for j in 1:k
            dist = Distances.evaluate(distance, X[:,i], centers[:,j])
            if dist < min_dist
                min_dist = dist
                best_cluster = j
            end
        end

        assignments[i] = best_cluster
    end

    # Check for NaN values in centers
    if any(isnan, centers)
        @warn "NaN values detected in cluster centers"
        return KMeansResult(centers, assignments, false)
    end

    return KMeansResult(centers, assignments, true)
end

"""
    kmeans_plusplus_init(X::AbstractMatrix{<:Real}, k::Int, distance::Distances.SemiMetric, rng::Random.AbstractRNG)

Initialize k-means cluster centers using the k-means++ method.

This method selects initial centers with probability proportional to their 
squared distance from existing centers, resulting in better initial clustering
than random selection.

# Data Format
- This function expects data with features in rows and observations in columns.
- Matrix X should be of size (n_features × n_samples).

# Arguments
- `X::AbstractMatrix{<:Real}`: Data matrix (n_features × n_samples)
- `k::Int`: Number of clusters
- `distance::Distances.SemiMetric`: Distance metric to use
- `rng::Random.AbstractRNG`: Random number generator

# Returns
- `::Matrix`: Initial cluster centers (n_features × k)
"""
function kmeans_plusplus_init(X::AbstractMatrix{T}, k::Int, 
                           distance::Distances.SemiMetric,
                           rng::Random.AbstractRNG) where {T<:Real}
    n_features, n_samples = size(X)
    centers = Matrix{T}(undef, n_features, k)

    # Choose first center randomly
    first_idx = rand(rng, 1:n_samples)
    centers[:,1] = X[:,first_idx]

    # Initialize distances
    min_dists = Vector{T}(undef, n_samples)
    for i in 1:n_samples
        min_dists[i] = Distances.evaluate(distance, X[:,i], centers[:,1])
    end

    # Choose remaining centers
    for c in 2:k
        # Convert distances to probabilities
        sum_dists = sum(min_dists)
        if sum_dists ≈ zero(T)
            # If all points are very close to existing centers, choose randomly
            next_idx = rand(rng, 1:n_samples)
        else
            probs = min_dists ./ sum_dists
            # Choose next center with probability proportional to distance
            next_idx = StatsBase.sample(rng, 1:n_samples, StatsBase.Weights(probs))
        end
        
        centers[:,c] = X[:,next_idx]

        # Update distances to nearest center
        for i in 1:n_samples
            dist = Distances.evaluate(distance, X[:,i], centers[:,c])
            if dist < min_dists[i]
                min_dists[i] = dist
            end
        end
    end

    return centers
end