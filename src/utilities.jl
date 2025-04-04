using CSV, DataFrames, StatsBase, Random

# ==========================================
# Data Preparation Utilities
# ==========================================

"""
    prepare_matrix(data; label_col=nothing, transpose=false)

Prepare data for kernel operations by converting to a matrix with observations in columns.

# Arguments
- `data`: A DataFrame, Matrix, or CSV file path
- `label_col`: Optional name of the label/target column (removed from features)
- `transpose`: Set to true if the data already has observations in columns

# Returns
- If label_col is provided:
  - `X`: Feature matrix (n_features × n_samples)
  - `y`: Vector of labels (length n_samples)
- Otherwise:
  - `X`: Feature matrix (n_features × n_samples)

# Examples
```julia
# From CSV file
X, y = prepare_matrix("data.csv", label_col="target")

# From DataFrame with label
df = DataFrame(x1=[1,2,3], x2=[4,5,6], y=[0,1,0])
X, y = prepare_matrix(df, label_col="y")
# X will be a 2×3 matrix with features in rows
# y will be [0,1,0]

# From DataFrame without label
df = DataFrame(x1=[1,2,3], x2=[4,5,6])
X = prepare_matrix(df)
# X will be a 2×3 matrix with features in rows

# From matrix in standard format (samples in rows)
data = [1 2 3; 4 5 6]  # 2 samples, 3 features each
X = prepare_matrix(data)
# X will be a 3×2 matrix with features in rows

# From matrix already in FlexKernel format (features in rows)
data = [1 4; 2 5; 3 6]  # 3 features, 2 samples
X = prepare_matrix(data, transpose=true)
# X will be unchanged, a 3×2 matrix
```
"""
function prepare_matrix(data; label_col=nothing, transpose=false)
    # Handle different input types
    if isa(data, AbstractString)
        # Load from CSV file
        df = CSV.read(data, DataFrame)
        return prepare_matrix(df; label_col=label_col, transpose=transpose)
        
    elseif isa(data, DataFrame)
        # Convert DataFrame to matrix, handling label column if specified
        if label_col !== nothing
            y = Vector(data[:, label_col])
            feature_cols = setdiff(names(data), [string(label_col)])
            X = Matrix{Float64}(data[:, feature_cols])
        else
            y = nothing
            X = Matrix{Float64}(data)
        end
        
        # Transpose for kernel operation (features in rows, observations in columns)
        if !transpose
            X = permutedims(X)
        end
        
        return label_col !== nothing ? (X, y) : X
        
    elseif isa(data, AbstractMatrix)
        # Handle matrix input
        if !transpose
            X = permutedims(data)
        else
            X = data
        end
        return X
        
    else
        throw(ArgumentError("Unsupported data type: $(typeof(data))"))
    end
end

"""
    to_dataframe(X; feature_names=nothing, transpose=true)

Convert a matrix to a DataFrame with observations in rows.

# Arguments
- `X`: Matrix of data
- `feature_names`: Optional vector of feature names
- `transpose`: Set to true if X has observations in columns (need to transpose)

# Returns
- `df`: DataFrame with observations in rows

# Examples
```julia
# Convert a FlexKernel-format matrix to DataFrame
X = rand(3, 100)  # 3 features, 100 samples
df = to_dataframe(X)
# df will have 100 rows and 3 columns

# Convert with custom feature names
X = rand(2, 50)  # 2 features, 50 samples
df = to_dataframe(X, feature_names=["height", "weight"])
# df will have columns named "height" and "weight"

# Convert a matrix that already has observations in rows
X = rand(100, 3)  # 100 samples, 3 features
df = to_dataframe(X, transpose=false)
# df will have 100 rows and 3 columns
```
"""
function to_dataframe(X; feature_names=nothing, transpose=true)
    if transpose
        X = permutedims(X)
    end
    
    n_features = size(X, 2)
    
    if feature_names === nothing
        feature_names = ["feature_$i" for i in 1:n_features]
    end
    
    return DataFrame(X, Symbol.(feature_names))
end

"""
    split_data(X, y; test_ratio=0.2, seed=nothing)

Split data into training and test sets.

# Arguments
- `X`: Feature matrix (n_features × n_samples)
- `y`: Target vector (length n_samples)
- `test_ratio`: Proportion of data to use for testing (default: 0.2)
- `seed`: Random seed for reproducibility

# Returns
- `X_train`: Training features (n_features × n_train)
- `y_train`: Training targets (length n_train)
- `X_test`: Test features (n_features × n_test)
- `y_test`: Test targets (length n_test)

# Examples
```julia
X = rand(5, 100)  # 5 features, 100 samples
y = rand(100)
X_train, y_train, X_test, y_test = split_data(X, y, test_ratio=0.25, seed=42)
# X_train will be 5×75, X_test will be 5×25
```
"""
function split_data(X, y; test_ratio=0.2, seed=nothing)
    # Set random seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end
    
    n_features, n_samples = size(X)
    n_test = round(Int, test_ratio * n_samples)
    n_train = n_samples - n_test
    
    # Shuffle indices
    indices = Random.shuffle(1:n_samples)
    train_indices = indices[1:n_train]
    test_indices = indices[n_train+1:end]
    
    # Split data
    X_train = X[:, train_indices]
    y_train = y[train_indices]
    X_test = X[:, test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_test, y_test
end

"""
    standardize_features(X; return_params=false)

Standardize features to have zero mean and unit variance.

# Arguments
- `X`: Feature matrix (n_features × n_samples)
- `return_params`: Whether to return standardization parameters (default: false)

# Returns
- If return_params is true:
  - `X_std`: Standardized features
  - `means`: Feature means
  - `stds`: Feature standard deviations
- Otherwise:
  - `X_std`: Standardized features

# Examples
```julia
X = rand(3, 100)  # 3 features, 100 samples
X_std = standardize_features(X)

# With parameters for later use
X_train = rand(3, 80)  # 3 features, 80 samples
X_test = rand(3, 20)   # 3 features, 20 samples
X_train_std, means, stds = standardize_features(X_train, return_params=true)
X_test_std = (X_test .- means) ./ stds
```
"""
function standardize_features(X; return_params=false)
    n_features, n_samples = size(X)
    
    # Compute means and standard deviations
    means = mean(X, dims=2)
    stds = std(X, dims=2)  # corrected=true is the default anyway

    # Handle zero standard deviations
    stds[stds .== 0] .= 1.0
    
    # Standardize
    X_std = (X .- means) ./ stds
    
    if return_params
        return X_std, vec(means), vec(stds)
    else
        return X_std
    end
end