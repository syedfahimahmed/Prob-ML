import numpy as np
import matplotlib.pyplot as plt
#import math
from scipy.special import gammaln

def t_distribution(x, v):
    
    num = np.exp(gammaln((v+1)/2))
    denom = np.sqrt(np.pi*v) * np.exp(gammaln(v/2)) * (1+x**2/v)**((v+1)/2)
    
    return num/denom

def std_gauss_distribution(x):
    
    return 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)


if __name__ == '__main__':
    
    v_values = [0.1, 1, 10, 100, 10**6]
    
    for v in v_values:
        x = np.linspace(-10, 10, 1000)
        y = t_distribution(x, v)
        plt.plot(x, y, label=f'v = {v}')
    
    
    plt.plot(x, std_gauss_distribution(x), label='Standard Gaussian Distribution')
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.title('Student t Distribution with Varying Degrees of Freedom vs Standard Gaussian Distribution')
    plt.legend()
    plt.show()
    plt.savefig('output_1.png')