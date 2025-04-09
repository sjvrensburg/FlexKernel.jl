# FlexKernel.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repository](https://img.shields.io/badge/GitHub-sjvrensburg/FlexKernel.jl-blue.svg)](https://github.com/sjvrensburg/FlexKernel.jl)

**FlexKernel.jl** is a Julia package for kernel methods with efficient approximations, designed for large-scale machine learning tasks. It provides custom kernel implementations, Nyström approximations for kernel matrices, mini-batch k-means clustering, and visualization tools, all optimized for performance and usability.

## Features

- **Custom Kernel**: `SincKernel` - A separable kernel based on the sinc function.
- **Nyström Approximation**: Efficiently approximates large kernel matrices for scalable computations.
- **Mini-batch K-means**: Fast clustering for landmark selection or standalone use.
- **Data Utilities**: Tools for preparing, standardizing, and splitting data.
- **Plotting Functions**: Visualization tools for kernels, landmarks, and Nyström approximation diagnostics.

**Data Format**: All functions expect matrices with features in rows and observations in columns (`n_features × n_samples`).

## Installation

FlexKernel.jl can be installed directly from its GitHub repository using Julia's package manager. Open a Julia REPL and run:

```julia
using Pkg
Pkg.add(url="https://github.com/sjvrensburg/FlexKernel.jl")
```

### Dependencies

FlexKernel.jl relies on the following Julia packages:
- `LinearAlgebra`
- `Distances`
- `Random`
- `StatsBase`
- `KernelFunctions`
- `CSV`
- `DataFrames`
- `Plots`

These will be automatically installed when you add FlexKernel.jl.

## Usage

Below are examples demonstrating the core functionality of FlexKernel.jl.

### 1. Using the SincKernel

The `SincKernel` is a custom kernel based on the sinc function, useful for capturing oscillatory patterns in data.

```julia
using FlexKernel

# Define a SincKernel with bandwidth 1.0
k = SincKernel(1.0)

# Generate sample data (features in rows, samples in columns)
X = randn(2, 100)  # 2 features, 100 samples
Y = randn(2, 50)   # 2 features, 50 samples

# Compute the kernel matrix
K = kernelmatrix(k, X, Y, obsdim=2)  # 100×50 matrix
```

### 2. Nyström Approximation

The Nyström method approximates a kernel matrix for large datasets, enabling efficient matrix-vector products.

```julia
using FlexKernel
using Random

# Generate data
X = randn(3, 1000)  # 3 features, 1000 samples
k = SincKernel(0.5)

# Compute landmarks using mini-batch k-means
Random.seed!(42)
landmarks = compute_nystrom_landmarks(X, 50, seed=42)

# Compute Nyström approximation
nystrom = compute_nystrom_approximation(X, landmarks, k)

# Multiply with a vector
v = randn(1000)
Kv = multiply(nystrom, v)  # Approximate K * v
```

### 3. Mini-batch K-means Clustering

Mini-batch k-means is provided for clustering or selecting Nyström landmarks.

```julia
using FlexKernel
using Random

# Generate data
X = randn(4, 500)  # 4 features, 500 samples

# Perform clustering
Random.seed!(123)
result = mini_batch_kmeans(X, 5, batch_size=50, seed=123)

# Access results
centers = result.centers      # 4×5 matrix of cluster centers
assignments = result.assignments  # Vector of cluster indices for each sample
println("Clustering successful: ", result.success)
```

### 4. Data Preparation and Splitting

Utilities are included to prepare data and split it into training and test sets.

```julia
using FlexKernel
using DataFrames
using Random

# From a DataFrame
df = DataFrame(x1=[1, 2, 3], x2=[4, 5, 6], y=[0, 1, 0])
X, y = prepare_matrix(df, label_col="y")  # X: 2×3, y: [0, 1, 0]

# Standardize features
X_std = standardize_features(X)

# Split into training and test sets
Random.seed!(42)
X_train, y_train, X_test, y_test = split_data(X_std, y, test_ratio=0.33, seed=42)
```

### 5. Plotting and Visualization

FlexKernel.jl includes plotting functions to visualize kernels, landmark distributions, and Nyström approximation accuracy. These rely on the `Plots.jl` package.

#### Plotting a 1D Kernel

Visualize the kernel function `k(x0, x)` for a range of `x` values.

```julia
using FlexKernel
using Plots

# Define a kernel
k = SincKernel(1.0)

# Plot the kernel centered at x0 = 0.0
plot_kernel_1d(k, 0.0, -5:0.1:5, title="Sinc Kernel", xlabel="x", ylabel="k(0, x)", label="Sinc")
```

#### Comparing Kernels

Compare multiple kernels on the same plot.

```julia
using FlexKernel
using KernelFunctions
using Plots

# Define kernels
k1 = SincKernel(1.0)
k2 = SqExponentialKernel()

# Compare kernels
compare_kernels_1d([k1, k2], 0.0, -5:0.1:5, labels=["Sinc", "RBF"], title="Kernel Comparison")
```

#### Visualizing a 2D Kernel

Create a heatmap of a kernel function over a 2D grid.

```julia
using FlexKernel
using Plots

# Define a kernel
k = SincKernel(0.5)

# Plot a 2D heatmap
plot_kernel_2d(k, [0.0, 0.0], -3:0.1:3, -3:0.1:3, title="Sinc Kernel Heatmap", xlabel="x1", ylabel="x2")
```

#### Plotting Landmarks

Visualize data points and Nyström landmarks in 2D, with optional clustering.

```julia
using FlexKernel
using Plots
using Random

# Generate data
Random.seed!(42)
X = randn(2, 100)  # 2 features, 100 samples
landmarks = compute_nystrom_landmarks(X, 5, seed=42)

# Plot data and landmarks
plot_landmarks_2d(X, landmarks, title="Landmark Distribution", color_by_cluster=true)
```

#### Analyzing Nyström Approximation

Plot the eigenvalues of the Nyström approximation versus the full kernel matrix to assess accuracy.

```julia
using FlexKernel
using Plots
using Random

# Generate data and compute Nyström approximation
Random.seed!(42)
X = randn(2, 50)  # 2 features, 50 samples
k = SincKernel(0.5)
landmarks = compute_nystrom_landmarks(X, 10, seed=42)
nystrom = compute_nystrom_approximation(X, landmarks, k)

# Plot eigenvalue comparison
plot_nystrom_accuracy(nystrom, X, k, compute_full=true, title="Nyström Approximation Accuracy")
```

### 6. Full Example: Kernel Regression with Nyström

A complete workflow combining kernel methods and Nyström approximation.

```julia
using FlexKernel
using LinearAlgebra
using Random

# Generate synthetic data
Random.seed!(42)
X = randn(2, 200)  # 2 features, 200 samples
y = sin.(X[1, :]) + cos.(X[2, :]) + 0.1 * randn(200)

# Split data
X_train, y_train, X_test, y_test = split_data(X, y, test_ratio=0.25, seed=42)

# Define kernel and compute Nyström approximation
k = SincKernel(0.5)
landmarks = compute_nystrom_landmarks(X_train, 20, seed=42)
nystrom = compute_nystrom_approximation(X_train, landmarks, k)

# Solve kernel regression (simplified)
K_train = nystrom.K_nm * nystrom.K_mm_inv * nystrom.K_nm'  # Approximate kernel matrix
alpha = K_train \ y_train  # Regression coefficients

# Predict on test set
K_test_train = multiply_validation(nystrom, X_test, ones(size(X_train, 2)), k)
y_pred = K_test_train * alpha

# Compute mean squared error
mse = mean((y_test - y_pred) .^ 2)
println("Test MSE: ", mse)
```

## Documentation

Full documentation for each function is available in the source code via Julia's docstrings. Access them in the REPL with:

```julia
?FlexKernel.SincKernel
?FlexKernel.compute_nystrom_approximation
?FlexKernel.mini_batch_kmeans
?FlexKernel.plot_kernel_1d
```

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, open an issue first to discuss your ideas.

- Repository: [https://github.com/sjvrensburg/FlexKernel.jl](https://github.com/sjvrensburg/FlexKernel.jl)
- Issues: [Report bugs or suggest features](https://github.com/sjvrensburg/FlexKernel.jl/issues)

## License

FlexKernel.jl is licensed under the MIT License. See the [LICENSE](https://github.com/sjvrensburg/FlexKernel.jl/blob/main/LICENSE) file for details.

```
Copyright (c) 2025 sjvrensburg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

This package builds on the excellent `KernelFunctions.jl` package and follows Julia's conventions for numerical computing. Thanks to the Julia community for their support and contributions to the ecosystem!

---

Happy kernel hacking with FlexKernel.jl!