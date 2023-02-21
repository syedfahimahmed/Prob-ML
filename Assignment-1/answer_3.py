import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def gaussian_likelihood(theta, x):
    
    mu, sigma = theta
    log_likelihood = -np.sum(np.log(1 / np.sqrt(2 * np.pi * sigma**2)) - (x - mu)**2 / (2 * sigma**2))
    return log_likelihood


def mle_gauss_dist(x):
    
    x_mean = np.mean(x)
    x_std = np.std(x)
    theta_init = [x_mean, x_std]
    result = minimize(gaussian_likelihood, theta_init, args=(x,), method='L-BFGS-B')
    mu_mle, sigma_mle = result.x
    return mu_mle, sigma_mle


def t_likelihood(theta, x):
    
    v = theta[0]
    log_likelihood = -np.sum(gammaln((v+1)/2) - gammaln(v/2) - np.log(np.sqrt(np.pi*v)) - (x+1)/2 * np.log(1 + (x**2)/v))
    return log_likelihood


def mle_t_dist(x):
    
    v_value = 1
    result = minimize(t_likelihood, [v_value], args=(x,), method='L-BFGS-B')
    v = result.x
    return v


def t_distribution(x, v):
    
    return np.exp(gammaln((v+1)/2) - gammaln(v/2) - np.log(np.sqrt(np.pi*v)) - (v+1)/2 * np.log(1 + (x**2)/v))


def std_gauss_distribution(x, mu, sigma):
    
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))



if __name__ == '__main__':
    
    data = np.random.normal(0, np.sqrt(2), 30)
    
    mu, sigma = mle_gauss_dist(data)
    v = mle_t_dist(data)
    

    x = np.linspace(-5, 5, 1000)
    plt.hist(data, bins=20, density=True, label='data')
    plt.plot(x, std_gauss_distribution(x, mu, sigma), label='gaussian distribution')
    plt.plot(x, t_distribution(x, v), label='student t-distribution')

    
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.title(' the estimated the Gaussian distribution density, student t density and the scatter data points')
    plt.legend()
    plt.show()
    plt.savefig('Output_4.png')

    plt.clf()

    # add 3 more samples
    data2 = np.append(data, [8, 9, 10])


    mu, sigma = mle_gauss_dist(data2)
    v = mle_t_dist(data2)

    x = np.linspace(-5, 11, 1000)
    plt.hist(data2, bins=20, density=True, label='data')
    plt.plot(x, std_gauss_distribution(x, mu, sigma), label='gaussian distribution')
    plt.plot(x, t_distribution(x, v), label='student t-distribution')

    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.title(' the estimated the Gaussian distribution density, student t density and the scatter data points (with noises)')
    plt.legend()
    plt.show()
    plt.savefig('Output_5.png')    