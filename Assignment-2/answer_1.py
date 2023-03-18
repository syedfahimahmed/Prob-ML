import numpy as np
import matplotlib.pyplot as plt

# Ground truth parameters
w0_true = -0.3
w1_true = 0.5

# Noise standard deviation
noise_std = 0.2

# Number of samples
num_samples = 20

# Generate input x and output y
np.random.seed(0)
x = np.random.uniform(-1, 1, num_samples)
y_true = w0_true + w1_true * x
y = y_true + np.random.normal(0, noise_std, num_samples)

# Hyperparameters
alpha = 2
beta = 25

# Prior distribution parameters
prior_mean = np.array([0, 0])
prior_cov = alpha * np.eye(2)

# Create a grid of points in the parameter space
w0_range = np.linspace(-1, 1, 100)
w1_range = np.linspace(-1, 1, 100)
W0, W1 = np.meshgrid(w0_range, w1_range)

# Evaluate the prior distribution for each point
W = np.stack([W0, W1], axis=-1)
prior_values = np.exp(-0.5 * np.sum((W - prior_mean) @ np.linalg.inv(prior_cov) * (W - prior_mean), axis=-1))


# Draw the heatmap
plt.imshow(prior_values, origin='lower', cmap='coolwarm', extent=(-1, 1, -1, 1))
plt.xlabel('w0')
plt.ylabel('w1')
plt.title('Heatmap of the prior p(w)')
plt.colorbar(label='Prior value')
plt.show()


# Sample 20 instances of w from the prior distribution
num_instances = 20
w_samples = np.random.multivariate_normal(prior_mean, prior_cov, num_instances)

# Plot the lines corresponding to each sampled w
x_range = np.linspace(-1, 1, 100)
plt.figure()

for i in range(num_instances):
    w0, w1 = w_samples[i]
    y_range = w0 + w1 * x_range
    plt.plot(x_range, y_range, alpha=0.5, color="blue")

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("20 lines sampled from the prior distribution")
plt.show()

# Bayesian linear regression - using only (x1, y1)
x1, y1 = x[0], y[0]
X1 = np.array([[1, x1]])
S0_inv = alpha * np.eye(2)
S_N1 = np.linalg.inv(S0_inv + beta * X1.T @ X1)
m_N1 = beta * S_N1 @ X1.T @ np.array([y1])

print(f"Posterior mean (m_N): {m_N1}")
print(f"Posterior covariance (S_N): \n{S_N1}")

posterior_values = np.exp(-0.5 * np.sum((W - m_N1) @ np.linalg.inv(S_N1) * (W - m_N1), axis=-1))


# Draw the heatmap
plt.imshow(posterior_values, origin='lower', cmap='coolwarm', extent=(-1, 1, -1, 1))
plt.xlabel('w0')
plt.ylabel('w1')
plt.title('Heatmap of the posterior p(w|x1, y1) with ground-truth')

# Plot the ground-truth values of w0 and w1
plt.scatter(w0_true, w1_true, color='red', marker='x', s=100, label='Ground-truth')
plt.legend()
plt.colorbar(label='Posterior value')
plt.show()

# Sample 20 instances of w from the posterior distribution
w_samples_posterior = np.random.multivariate_normal(m_N1, S_N1, num_instances)

# Plot the lines corresponding to each sampled w
plt.figure()

for i in range(num_instances):
    w0, w1 = w_samples_posterior[i]
    y_range = w0 + w1 * x_range
    plt.plot(x_range, y_range, alpha=0.5, color="blue")

# Plot the data point (x1, y1)
plt.scatter(x1, y1, color='red', marker='o', s=100, label='(x1, y1)')

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("20 lines sampled from the posterior distribution and data point (x1, y1)")
plt.show()

# ques_3
# Bayesian linear regression - using (x1, y1) and (x2, y2)
x2, y2 = x[1], y[1]
X2 = np.array([[1, x1], [1, x2]])
S_N2 = np.linalg.inv(S0_inv + beta * X2.T @ X2)
m_N2 = beta * S_N2 @ X2.T @ np.array([y1, y2])

print(f"Posterior mean (m_N): {m_N2}")
print(f"Posterior covariance (S_N): \n{S_N2}")


posterior_values_2 = np.exp(-0.5 * np.sum((W - m_N2) @ np.linalg.inv(S_N2) * (W - m_N2), axis=-1))


# Draw the heatmap
plt.imshow(posterior_values_2, origin='lower', cmap='coolwarm', extent=(-1, 1, -1, 1))
plt.xlabel('w0')
plt.ylabel('w1')
plt.title('Heatmap of the posterior p(w|x1, y1, x2, y2) with ground-truth')

# Plot the ground-truth values of w0 and w1
plt.scatter(w0_true, w1_true, color='red', marker='x', s=100, label='Ground-truth')
plt.legend()

plt.colorbar(label='Posterior value')
plt.show()

# Sample 20 instances of w from the posterior distribution
w_samples_posterior_2 = np.random.multivariate_normal(m_N2, S_N2, num_instances)

# Plot the lines corresponding to each sampled w
plt.figure()

for i in range(num_instances):
    w0, w1 = w_samples_posterior_2[i]
    y_range = w0 + w1 * x_range
    plt.plot(x_range, y_range, alpha=0.5, color="blue")

# Plot the data points (x1, y1) and (x2, y2)
plt.scatter([x1, x2], [y1, y2], color='red', marker='o', s=100, label='(x1, y1) & (x2, y2)')

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("20 lines sampled from the posterior distribution and data points (x1, y1) & (x2, y2)")
plt.show()

# Bayesian linear regression - using the first five data points
X5 = np.hstack([np.ones((5, 1)), x[:5].reshape(-1, 1)])
y5 = y[:5]
S_N5 = np.linalg.inv(S0_inv + beta * X5.T @ X5)
m_N5 = beta * S_N5 @ X5.T @ y5

posterior_values_5 = np.exp(-0.5 * np.sum((W - m_N5) @ np.linalg.inv(S_N5) * (W - m_N5), axis=-1))

# Draw the heatmap
plt.imshow(posterior_values_5, origin='lower', cmap='coolwarm', extent=(-1, 1, -1, 1))
plt.xlabel('w0')
plt.ylabel('w1')
plt.title('Heatmap of the posterior p(w|x1..x5, y1..y5) with ground-truth')

# Plot the ground-truth values of w0 and w1
plt.scatter(w0_true, w1_true, color='red', marker='x', s=100, label='Ground-truth')
plt.legend()

plt.colorbar(label='Posterior value')
plt.show()

# Sample 20 instances of w from the posterior distribution
w_samples_posterior_5 = np.random.multivariate_normal(m_N5, S_N5, num_instances)

# Plot the lines corresponding to each sampled w
plt.figure()

for i in range(num_instances):
    w0, w1 = w_samples_posterior_5[i]
    y_range = w0 + w1 * x_range
    plt.plot(x_range, y_range, alpha=0.5, color="blue")

# Plot the data points (x1, y1), ... , (x5, y5)
plt.scatter(x[:5], y[:5], color='red', marker='o', s=100, label='(x1..x5, y1..y5)')

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("20 lines sampled from the posterior distribution and data points (x1..x5, y1..y5)")
plt.show()


# Bayesian linear regression - using all 20 data points
X_all = np.hstack([np.ones((20, 1)), x.reshape(-1, 1)])
y_all = y
S_N_all = np.linalg.inv(S0_inv + beta * X_all.T @ X_all)
m_N_all = beta * S_N_all @ X_all.T @ y_all

posterior_values_all = np.exp(-0.5 * np.sum((W - m_N_all) @ np.linalg.inv(S_N_all) * (W - m_N_all), axis=-1))

# Draw the heatmap
plt.imshow(posterior_values_all, origin='lower', cmap='coolwarm', extent=(-1, 1, -1, 1))
plt.xlabel('w0')
plt.ylabel('w1')
plt.title('Heatmap of the posterior p(w|x1..x20, y1..y20) with ground-truth')

# Plot the ground-truth values of w0 and w1
plt.scatter(w0_true, w1_true, color='red', marker='x', s=100, label='Ground-truth')
plt.legend()

plt.colorbar(label='Posterior value')
plt.show()

# Sample 20 instances of w from the posterior distribution
w_samples_posterior_all = np.random.multivariate_normal(m_N_all, S_N_all, num_instances)

# Plot the lines corresponding to each sampled w
plt.figure()

for i in range(num_instances):
    w0, w1 = w_samples_posterior_all[i]
    y_range = w0 + w1 * x_range
    plt.plot(x_range, y_range, alpha=0.5, color="blue")

# Plot the data points (x1, y1), ... , (x20, y20)
plt.scatter(x, y, color='red', marker='o', s=100, label='(x1..x20, y1..y20)')

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("20 lines sampled from the posterior distribution and data points (x1..x20, y1..y20)")
plt.show()