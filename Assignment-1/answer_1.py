import numpy as np
import matplotlib.pyplot as plt
#import math
from scipy.special import gammaln

def t_distribution(x, v):
    
    return np.exp(gammaln((v+1)/2) - gammaln(v/2) - np.log(np.sqrt(np.pi*v)) - (v+1)/2 * np.log(1 + (x**2)/v))


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
    plt.savefig('ques_1_output_1.png')
    plt.show()