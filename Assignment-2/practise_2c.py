import numpy as np
import pandas as pd
from scipy.stats import norm

# Define probit function and its derivative
def probit(z):
    return norm.cdf(z)
def probit_deriv(z):
    return norm.pdf(z)

class ProbitRegressionNR:
    def __init__(self, max_iter=100, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        
    def fit(self, X, y, weights="zeros"):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        if weights == "zeros": # Initialize weights to zero
            self.weights = np.zeros(X.shape[1])
        else: # Initialize weights with standard Gaussian
            self.weights = np.random.normal(size=X.shape[1])
        
        # Define negative log-posterior function
        def neg_log_posterior(w):
            y_pred = probit(X @ w)
            return -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred)) - 0.5*np.sum(w**2)
        
        # Define gradient of negative log-posterior function
        def neg_log_posterior_grad(w):
            y_pred = probit(X @ w)
            return X.T @ (y_pred - y) - w
        
        # Define Hessian of negative log-posterior function
        def neg_log_posterior_hess(w):
            y_pred = probit(X @ w)
            W = np.diag(probit_deriv(X @ w))
            return X.T @ W @ X + np.eye(X.shape[1])
        
        # Update weights using Newton-Raphson scheme
        for i in range(self.max_iter):
            old_weights = self.weights
            grad = neg_log_posterior_grad(self.weights)
            hess = neg_log_posterior_hess(self.weights)
            self.weights = old_weights - np.linalg.inv(hess) @ grad
            if np.linalg.norm(self.weights - old_weights) < self.tol:
                break
                
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Compute predicted probabilities
        y_pred = probit(X @ self.weights)
        
        # Convert probabilities to binary labels
        return (y_pred >= 0.5).astype(int)
    
train_df = pd.read_csv("Assignment-2/bank-note/train.csv")
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

test_df = pd.read_csv("Assignment-2/bank-note/test.csv")
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

pr_nr = ProbitRegressionNR()
pr_nr.fit(X_train, y_train)

train_pred = pr_nr.predict(X_train)
train_acc = np.mean(train_pred == y_train)
print('Training accuracy for zero weights:', train_acc)

test_pred = pr_nr.predict(X_test)
test_acc = np.mean(test_pred == y_test)
print('Test accuracy for zero weights:', test_acc)

pr_nr_rand = ProbitRegressionNR()
pr_nr_rand.fit(X_train, y_train, weights="random")

train_rand_pred = pr_nr_rand.predict(X_train)
train_rand_acc = np.mean(train_rand_pred == y_train)
print('Training accuracy for random weights:', train_rand_acc)

test_rand_pred = pr_nr_rand.predict(X_test)
test_rand_acc = np.mean(test_rand_pred == y_test)
print('Test accuracy for random weights:', test_rand_acc)

# Save accuracies to file
with open('accuracy_probit_reg_nr.txt', 'w') as f:
    f.write('Probit Regression (Newton-Raphson scheme implemented)\n')
    f.write(f'Training accuracy for zero weights:: {train_acc:.4f}\n')
    f.write(f'Test accuracy for zero weights:: {test_acc:.4f}\n\n')
    f.write(f'Training accuracy for random weights:: {train_rand_acc:.4f}\n')
    f.write(f'Test accuracy for random weights:: {test_rand_acc:.4f}\n\n')