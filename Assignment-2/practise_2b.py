import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

# Define probit function
def probit(z):
    return norm.cdf(z)

class ProbitRegression:
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
        def neg_log_posterior(w, epsilon=1e-7):
            y_pred = probit(X @ w)
            return -np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon)) - 0.5 * np.sum(w**2)
        
        # Define gradient of negative log-posterior function
        def neg_log_posterior_grad(w):
            y_pred = probit(X @ w)
            return X.T @ (y_pred - y) - w
        
        # Optimize negative log-posterior function using L-BFGS
        result = minimize(neg_log_posterior, self.weights, method='L-BFGS-B', jac=neg_log_posterior_grad, options={'maxiter': self.max_iter, 'gtol': self.tol})
        self.weights = result.x
                
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

pr = ProbitRegression()
pr.fit(X_train, y_train)

train_pred = pr.predict(X_train)
train_acc = np.mean(train_pred == y_train)
print('Training accuracy for zero weights:', train_acc)

test_pred = pr.predict(X_test)
test_acc = np.mean(test_pred == y_test)
print('Test accuracy for zero weights:', test_acc)

pr_rand = ProbitRegression()
pr_rand.fit(X_train, y_train, weights="random")

train_rand_pred = pr_rand.predict(X_train)
train_rand_acc = np.mean(train_rand_pred == y_train)
print('Training accuracy for random weights:', train_rand_acc)

test_rand_pred = pr_rand.predict(X_test)
test_rand_acc = np.mean(test_rand_pred == y_test)
print('Test accuracy for random weights:', test_rand_acc)

# Save accuracies to file
with open('accuracy_probit_reg.txt', 'w') as f:
    f.write('Probit Regression\n')
    f.write(f'Training accuracy for zero weights:: {train_acc:.4f}\n')
    f.write(f'Test accuracy for zero weights:: {test_acc:.4f}\n\n')
    f.write(f'Training accuracy for random weights:: {train_rand_acc:.4f}\n')
    f.write(f'Test accuracy for random weights:: {test_rand_acc:.4f}\n\n')