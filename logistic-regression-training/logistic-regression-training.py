import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X)
    y = np.array(y)
    
    N, D = X.shape
    
    # Initialize parameters
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        # Forward pass
        z = X @ w + b
        p = sigmoid(z)
        
        # Gradients
        dw = (X.T @ (p - y)) / N
        db = np.mean(p - y)
        
        # Update
        w -= lr * dw
        b -= lr * db
    
    return w, b