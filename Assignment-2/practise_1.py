import numpy as np
import matplotlib.pyplot as plt

def generate_data(w0_true, w1_true, n_samples, x_range, noise_std):
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    y = w0_true + w1_true * x + np.random.normal(0, noise_std, n_samples)
    return x, y

def prior_heatmap_and_lines(alpha, W, prior_mean, prior_cov):

    prior_values = np.exp(-0.5 * np.sum((W - prior_mean) @ np.linalg.inv(prior_cov) * (W - prior_mean), axis=-1))

    plt.imshow(prior_values, origin='lower', cmap='coolwarm', extent=(-1, 1, -1, 1))
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.title('Heatmap of the prior p(w)')
    plt.colorbar(label='Prior value')
    plt.savefig(f'Prior.png')
    plt.show()
    
    num_instances = 20
    w_samples = np.random.multivariate_normal(prior_mean, prior_cov, num_instances)
    
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
    plt.savefig(f'Line_Prior.png')
    plt.show()

def posterior_distribution(X, y, alpha, beta):
    S0_inv = alpha * np.eye(X.shape[1])
    S_N = np.linalg.inv(S0_inv + beta * X.T @ X)
    m_N = beta * S_N @ X.T @ y
    
    print(f"Posterior mean (m_N): {m_N}")
    print(f"Posterior covariance (S_N): \n{S_N}")
    return m_N, S_N

def posterior_heatmap_and_lines(m_N, S_N, x, y, W, w0_true, w1_true, x_range, num_instances, data_points_to_draw=None):

    posterior_values = np.exp(-0.5 * np.sum((W - m_N) @ np.linalg.inv(S_N) * (W - m_N), axis=-1))

    plt.scatter(w0_true, w1_true, color='red', marker='x', s=100, label='Ground-truth')
    plt.imshow(posterior_values, origin='lower', cmap='coolwarm', extent=(-1, 1, -1, 1))
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.title('Heatmap of the posterior p(w|x, y) with ground-truth')

    plt.legend()

    plt.colorbar(label='Posterior value')
    plt.savefig(f'Posterior value for (x1..x{data_points_to_draw}, y1..y{data_points_to_draw}).png')
    plt.show()

    w_samples_posterior = np.random.multivariate_normal(m_N, S_N, num_instances)

    plt.figure()

    for i in range(num_instances):
        w0, w1 = w_samples_posterior[i]
        y_range = w0 + w1 * x_range
        plt.plot(x_range, y_range, alpha=0.5, color="blue")

    if data_points_to_draw:
        plt.scatter(x[:data_points_to_draw], y[:data_points_to_draw], color='red', marker='o', s=100, label=f'(x1..x{data_points_to_draw}, y1..y{data_points_to_draw})')
        plt.legend()

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"20 lines sampled from the posterior distribution and data points (x1..x{data_points_to_draw}, y1..y{data_points_to_draw})")
    plt.savefig(f'20 lines sampled from the posterior distribution and data points (x1..x{data_points_to_draw}, y1..y{data_points_to_draw}).png')
    plt.show()


# Set parameters
w0_true, w1_true = -0.3, 0.5
n_samples = 20
x_range = (-1, 1)
noise_std = 0.2
alpha = 2
beta = 25

# Generate data
x, y = generate_data(w0_true, w1_true, n_samples, x_range, noise_std)

w0_range = np.linspace(-1, 1, 100)
w1_range = np.linspace(-1, 1, 100)

W0, W1 = np.meshgrid(w0_range, w1_range)
W = np.stack([W0, W1], axis=-1)

# Prior parameters
prior_mean = np.zeros(2)
prior_cov = 2 * np.identity(2)

# Task 1: Prior heatmap and lines from prior distribution
prior_heatmap_and_lines(alpha, W, prior_mean, prior_cov)

X = np.hstack([np.ones((n_samples, 1)), x.reshape(-1, 1)])
x_range = np.linspace(-1, 1, 100)
num_instances = 20

# Task 2: Posterior heatmap and lines given (x1, y1)
m_N_1, S_N_1 = posterior_distribution(X[:1], y[:1], alpha, beta)
posterior_heatmap_and_lines(m_N_1, S_N_1, x, y, W, w0_true, w1_true, x_range, num_instances, data_points_to_draw=1)

# Task 3: Posterior heatmap and lines given (x1, y1), (x2, y2)
m_N_2, S_N_2 = posterior_distribution(X[:2], y[:2], alpha, beta)
posterior_heatmap_and_lines(m_N_2, S_N_2, x, y, W, w0_true, w1_true, x_range, num_instances, data_points_to_draw=2)

# Task 4: Posterior heatmap and lines given (x1, y1), ... , (x5, y5)
m_N_5, S_N_5 = posterior_distribution(X[:5], y[:5], alpha, beta)
posterior_heatmap_and_lines(m_N_5, S_N_5, x, y, W, w0_true, w1_true, x_range, num_instances, data_points_to_draw=5)


# Task 5: Posterior heatmap and lines given (x1, y1), ... , (x20, y20)
m_N_20, S_N_20 = posterior_distribution(X, y, alpha, beta)
posterior_heatmap_and_lines(m_N_20, S_N_20, x, y, W, w0_true, w1_true, x_range, num_instances, data_points_to_draw=20)