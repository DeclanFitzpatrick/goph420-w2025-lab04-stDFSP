import numpy as np
import matplotlib.pyplot as plt
from lab_04.regression import (multi_regress)


def main():
    data = np.loadtxt("data/M_data1.txt")

    t = data[:,0]  # dependent data
    m_data = data[:,1]  # independent data

    k_24 = np.argwhere(t < 24)[-1].item()

    m = np.linspace(-0.5, 1, 9)
    n = np.zeros_like(m)  # array of zeros same size as m

    for i, mm in enumerate(m):  # loop through m and count values in m_data that are greater than or equal to mm
        n[i] = np.count_nonzero(m_data[:k_24] >= mm)  # count occurrences
    print(n)
    y = np.log10(n)
    Z = np.vstack((np.ones_like(m), m)).T  # design matrix

    a, e, rsq = multi_regress(y, Z)  # use least square regression to find coeffs, residuals, and R^2

    print(f'a: {a}')
    print(f' e: {e}')
    print(f' rsq: {rsq}')

    Log1 = Z @ a  # compute predicted values


    plt.plot(m, y, 'o')
    plt.plot(m, Log1, 'o')
    plt.xlabel('m')
    plt.ylabel('log10(n)')
    plt.title('log10(n) vs m')
    plt.show()


if __name__ == "__main__":
    main()