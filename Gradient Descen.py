import numpy as np

# Generate random data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Perform gradient descent for linear regression
eta = 0.1  # learning rate
n_iterations = 1000
theta = np.random.randn(2, 1)
X_b = np.c_[np.ones((100, 1)), X]
for iteration in range(n_iterations):
    gradients = 2 / 100 * X_b.T.dot(X_b.dot(theta) - y)
    theta -= eta * gradients

print("Optimal parameters (Linear Regression with Gradient Descent):")
print(theta)
