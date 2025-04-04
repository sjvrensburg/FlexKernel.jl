using Test
using FlexKernel
using KernelFunctions
using LinearAlgebra
using Random
using Distances
using StatsBase

@testset "FlexKernel" begin
    
    @testset "SincKernel" begin
        # Test construction
        @test SincKernel(1.0) isa SincKernel
        @test_throws ArgumentError SincKernel(-1.0)
        
        # Test type parameterization
        k1 = SincKernel(1.0)
        k2 = SincKernel(1.0f0)
        @test typeof(k1.bandwidth) === Float64
        @test typeof(k2.bandwidth) === Float32
        
        # Test sinc_value function
        @test sinc_value(0.0) ≈ 1.0
        @test sinc_value(1.0) ≈ sin(π*1.0)/(π*1.0)
        @test sinc_value(0.5) ≈ sin(π*0.5)/(π*0.5)
        
        # Test point evaluation
        k = SincKernel(1.0)
        x = [1.0, 2.0]
        y = [1.0, 2.0]
        @test k(x, y) ≈ 1.0  # Same point should give 1.0
        
        # Test dimension mismatch
        @test_throws DimensionMismatch k([1.0], [1.0, 2.0])
        
        # Test correct calculation for different points
        z = [1.5, 2.5]
        expected = sinc_value((x[1]-z[1])/1.0) * sinc_value((x[2]-z[2])/1.0)
        @test k(x, z) ≈ expected
        
        # Test matrix computation
        X = randn(2, 5)  # 2 features, 5 samples
        Y = randn(2, 3)  # 2 features, 3 samples
        K = kernelmatrix(k, X, Y, obsdim=2)
        @test size(K) == (5, 3)
        
        # Test consistency between point and matrix evaluation
        for i in 1:5, j in 1:3
            @test K[i, j] ≈ k(X[:,i], Y[:,j])
        end
        
        # Test diagonal
        K_diag = kernelmatrix_diag(k, X, obsdim=2)
        @test length(K_diag) == 5
        @test all(K_diag .≈ 1.0)  # Diagonal should be all 1.0
    end
    
    @testset "Nyström Approximation" begin
        # Create test data
        d = 2
        n = 50
        m = 10
        X = randn(d, n)  # d features, n samples
        kernel = SqExponentialKernel()
        
        # Test landmark computation
        landmarks = compute_nystrom_landmarks(X, m, seed=42)
        @test size(landmarks) == (d, m)
        
        # Test with more landmarks than points
        @test_logs (:warn, r"Number of landmarks exceeds") compute_nystrom_landmarks(X, n+10)
        
        # Test approximation computation
        nystrom = compute_nystrom_approximation(X, landmarks, kernel)
        @test nystrom isa NystromApproximation
        @test size(nystrom.landmarks) == (d, m)
        @test size(nystrom.K_nm) == (n, m)
        @test size(nystrom.K_mm_inv) == (m, m)
        
        # Test matrix-vector product
        v = randn(n)
        Kv = multiply(nystrom, v)
        @test length(Kv) == n
        
        # Test dimension mismatch
        @test_throws DimensionMismatch multiply(nystrom, randn(n+1))
        
        # Test projection
        W = randn(n, 3)
        Kv_proj = multiply_with_projection(nystrom, v, W)
        @test length(Kv_proj) == n
        
        # Test validation
        X_val = randn(d, 5)  # d features, 5 validation samples
        Kval_v = multiply_validation(nystrom, X_val, v, kernel)
        @test length(Kval_v) == 5
        
        # Test dimension mismatch in validation
        @test_throws DimensionMismatch multiply_validation(nystrom, randn(d+1, 5), v, kernel)
    end
    
    @testset "K-means" begin
        # Create clustered data
        function generate_clusters(n_clusters, n_per_cluster, dim, scale=0.5)
            Random.seed!(42)
            # Generate data with features in columns (d × n format)
            data = zeros(dim, n_clusters * n_per_cluster)
            labels = zeros(Int, n_clusters * n_per_cluster)
            
            for i in 1:n_clusters
                center = 3 * randn(dim)
                idx_start = (i-1) * n_per_cluster + 1
                idx_end = i * n_per_cluster
                
                for j in idx_start:idx_end
                    data[:, j] = center + scale * randn(dim)
                    labels[j] = i
                end
            end
            
            return data, labels
        end
        
        data, true_labels = generate_clusters(3, 30, 2)
        
        # Test basic k-means
        result = mini_batch_kmeans(data, 3, seed=42)
        @test result isa KMeansResult
        @test result.success
        @test size(result.centers) == (2, 3)  # 2 features, 3 clusters
        @test length(result.assignments) == 90
        
        # Test with invalid inputs
        @test_throws ArgumentError mini_batch_kmeans(data, 0)
        @test_throws ArgumentError mini_batch_kmeans(data, 3, batch_size=0)
        @test_throws ArgumentError mini_batch_kmeans(data, 3, max_iterations=0)
        
        # Test with more clusters than data points
        small_data = data[:, 1:2]  # Only 2 samples
        @test_logs (:warn, r"Number of samples less than number of clusters") mini_batch_kmeans(small_data, 3)
        
        # Test with custom distance function
        result_manhattan = mini_batch_kmeans(data, 3, distance=Distances.Cityblock())
        @test result_manhattan.success
        
        # Test early convergence with tight tolerance
        result_converge = mini_batch_kmeans(data, 3, tol=1e-1, max_iterations=1000)
        @test result_converge.success
    end
    
    @testset "Type stability and flexibility" begin
        # Test with Float32
        X_f32 = randn(Float32, 2, 20)  # 2 features, 20 samples
        k_f32 = SincKernel(1.0f0)
        
        K_f32 = kernelmatrix(k_f32, X_f32, X_f32, obsdim=2)
        @test eltype(K_f32) == Float32
        
        # Test with mixed types (should promote to Float64)
        X_f64 = randn(2, 20)  # Float64 data
        K_mixed = kernelmatrix(k_f32, X_f64, X_f64, obsdim=2)
        @test eltype(K_mixed) == Float64
        
        # Test Nyström with Float32
        landmarks_f32 = compute_nystrom_landmarks(X_f32, 5)
        @test eltype(landmarks_f32) == Float32
        
        nystrom_f32 = compute_nystrom_approximation(X_f32, landmarks_f32, k_f32)
        @test eltype(nystrom_f32.K_nm) == Float32
        @test eltype(nystrom_f32.K_mm_inv) == Float32
        
        # Test K-means with Float32
        data_f32 = randn(Float32, 2, 30)  # 2 features, 30 samples
        result_f32 = mini_batch_kmeans(data_f32, 3)
        @test eltype(result_f32.centers) == Float32
    end
    
    @testset "Integration" begin
        # Test the full pipeline from data to kernel to Nyström to prediction
        d = 2  # 2 features
        n_train = 40  # 40 training samples
        n_test = 10  # 10 test samples
        m = 8  # 8 landmarks
        
        # Generate data
        Random.seed!(42)
        X_train = randn(d, n_train)  # d features, n_train samples
        y_train = sin.(X_train[1,:]) .* cos.(X_train[2,:])
        X_test = randn(d, n_test)  # d features, n_test samples
        y_test = sin.(X_test[1,:]) .* cos.(X_test[2,:])
        
        # Create kernel
        kernel = with_lengthscale(SqExponentialKernel(), 1.0)
        
        # Compute Nyström approximation
        landmarks = compute_nystrom_landmarks(X_train, m)
        nystrom = compute_nystrom_approximation(X_train, landmarks, kernel)
        
        # Simple kernel ridge regression
        lambda = 1e-3
        K = kernelmatrix(kernel, X_train, X_train, obsdim=2)
        alpha_exact = (K + lambda * I) \ y_train
        
        # Use Nyström for prediction
        K_test = kernelmatrix(kernel, X_test, X_train, obsdim=2)
        y_pred_exact = K_test * alpha_exact
        
        # Nyström approximation for test predictions
        K_test_nystrom = multiply_validation(nystrom, X_test, alpha_exact, kernel)
        
        # Compare predictions
        rmse = sqrt(mean((y_pred_exact - K_test_nystrom).^2))
        @test rmse < 0.1  # Allow small error due to approximation
    end
end