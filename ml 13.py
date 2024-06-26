import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal

# Step 1: Generate some synthetic data
np.random.seed(42)
n_samples = 300
X, _ = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.0, random_state=42)

# Step 2: Initialize the parameters
def initialize_parameters(X, n_clusters):
    n_samples, n_features = X.shape
    means = X[np.random.choice(n_samples, n_clusters, False)]
    covariances = [np.eye(n_features) for _ in range(n_clusters)]
    weights = np.ones(n_clusters) / n_clusters
    return means, covariances, weights

# Step 3: E-step: calculate responsibilities
def e_step(X, means, covariances, weights):
    n_samples, n_clusters = X.shape[0], len(means)
    responsibilities = np.zeros((n_samples, n_clusters))
    
    for k in range(n_clusters):
        responsibilities[:, k] = weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
    
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

# Step 4: M-step: update parameters
def m_step(X, responsibilities):
    n_samples, n_features = X.shape
    n_clusters = responsibilities.shape[1]
    
    Nk = responsibilities.sum(axis=0)
    means = np.dot(responsibilities.T, X) / Nk[:, None]
    covariances = []
    
    for k in range(n_clusters):
        diff = X - means[k]
        covariances.append(np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k])
    
    weights = Nk / n_samples
    return means, covariances, weights

# Step 5: Log likelihood
def log_likelihood(X, means, covariances, weights):
    n_samples = X.shape[0]
    likelihood = np.zeros(n_samples)
    
    for k in range(len(means)):
        likelihood += weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
    
    return np.sum(np.log(likelihood))

# Step 6: EM algorithm
def em_algorithm(X, n_clusters, max_iter=100, tol=1e-6):
    means, covariances, weights = initialize_parameters(X, n_clusters)
    log_likelihoods = []
    
    for i in range(max_iter):
        responsibilities = e_step(X, means, covariances, weights)
        means, covariances, weights = m_step(X, responsibilities)
        log_likelihoods.append(log_likelihood(X, means, covariances, weights))
        
        if i > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break
    
    return means, covariances, weights, responsibilities, log_likelihoods

# Run the EM algorithm
n_clusters = 2
means, covariances, weights, responsibilities, log_likelihoods = em_algorithm(X, n_clusters)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=responsibilities.argmax(axis=1), s=40, cmap='viridis')
plt.scatter(means[:, 0], means[:, 1], c='red', s=100, marker='x')
plt.title('EM Algorithm for Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plot the log likelihood
plt.plot(log_likelihoods)
plt.title('Log Likelihood')
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood')
plt.show()
