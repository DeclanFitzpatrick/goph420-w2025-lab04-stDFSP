import numpy as np
import matplotlib.pyplot as plt

from lab_04.regression import (multi_regress)

def main():
    data = np.loadtxt("data/M_data1.txt")

    t = data[:,0] # enter dependent variable data (or load from a file)
    m_data = data[:,1] # enter independent variable data (or load from a file)

    k_24 = np.argwhere(t < 24)[-1]
    print(k_24)
    print(t[k_24])


    m = np.linspace(-0.5, 1.5, 9)
    n = np.zeros_like(m)

    for i, mm in enumerate(m):
        n[i] = np.count_nonzero(m_data >= mm)

    y = np.log10(n)
    Z = np.vstack((np.ones_like(m), m)).T

    a, e, rsq = multi_regress(y, Z)

    print(a)
    print(e)
    print(rsq)

    plt.plot(m, y, 'o')
    plt.xlabel('m')
    plt.ylabel('log10(n)')
    plt.title('log10(n) vs m')
    plt.show()

if __name__ == "__main__":
    main()