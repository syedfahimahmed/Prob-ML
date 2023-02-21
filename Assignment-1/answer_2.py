import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt


def beta_pdf(x, a, b):
    
    beta = np.exp(gammaln(a) + gammaln(b) - gammaln(a+b))
    return x**(a-1) * (1-x)**(b-1) / beta



if __name__ == '__main__':
    
    a = [1, 5, 10]
    b = [1, 5, 10]

    x = np.linspace(0, 1, 1000)

    for i in range(len(a)):
        plt.plot(x, beta_pdf(x, a[i], b[i]), label=f'a = {a[i]}, b = {b[i]}')

    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.title('Density plots of Beta distributions for (1,1), (5,5) and (10,10)')
    plt.legend()
    plt.savefig('output_2.png')
    plt.show()
    
    a = [1, 5, 10]
    b = [2, 6, 11]

    plt.clf()

    for i in range(len(a)):
        plt.plot(x, beta_pdf(x, a[i], b[i]), label=f'a = {a[i]}, b = {b[i]}')

    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.title('Density plots of Beta distributions for (1,2), (5,6) and (10,11)')
    plt.legend()
    plt.savefig('output_3.png')
    plt.show()