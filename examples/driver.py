import numpy as np
import matplotlib.pyplot as plt
from lab_04.regression import (multi_regress)


def main():
    data = np.loadtxt("../data/M_data1.txt")

    t = data[:,0]  # dependent data
    m_data = data[:,1]  # independent data

    k_24 = np.argwhere(t < 24)[-1]  # find indices when t is less than 24 and select last index
    print(k_24)
    print(t[k_24])

    m = np.linspace(-0.5, 1.5, 9)
    n = np.zeros_like(m)  # array of zeros same size as m

    for i, mm in enumerate(m):  # loop through m and count values in m_data that are greater than or equal to mm
        n[i] = np.count_nonzero(m_data >= mm)  # count occurrences

    y = np.log10(n)
    Z = np.vstack((np.ones_like(m), m)).T  # design matrix

    a, e, rsq = multi_regress(y, Z)  # use least square regression to find coeffs, residuals, and R^2

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