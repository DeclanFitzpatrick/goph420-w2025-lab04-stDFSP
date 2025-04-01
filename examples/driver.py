import numpy as np
import matplotlib.pyplot as plt
from lab_04.regression import (multi_regress)


def main():
    data = np.loadtxt("../data/M_data1.txt")

    t = data[:,0]  # dependent data
    m_data = data[:,1]  # independent data

    k_34 = int(np.argwhere(t < 35)[-1].item())  # to extract scalar value and not return array

    m = np.linspace(-0.5, 1, 9)
    n = np.zeros_like(m)  # array of zeros same size as m

    for i, mm in enumerate(m):  # loop through m and count values in m_data that are greater than or equal to mm
        n[i] = np.count_nonzero(m_data[:k_34] >= mm)  # count occurrences
    print(n)
    y = np.log10(n)
    Z = np.vstack((np.ones_like(m), m)).T  # design matrix

    a, e, rsq = multi_regress(y, Z)  # use least square regression to find coeffs, residuals, and R^2

    print(f'a: {a}')
    print(f' e: {e}')
    print(f' rsq: {rsq}')

    Log1 = Z @ a  # compute predicted values

    plt.plot(m, y, 'o', label="data between 0 and 34")
    plt.plot(m, Log1, '-', label='fit 1')

    k_46 = int(np.argwhere(t < 46)[-1].item())

    m = np.linspace(-0.5, 1, 9)
    n = np.zeros_like(m)  # array of zeros same size as m

    for i, mm in enumerate(m):  # loop through m and count values in m_data that are greater than or equal to mm
        n[i] = np.count_nonzero(m_data[k_34:k_46] >= mm)  # count occurrences
    print(n)
    y = np.log10(n)
    Z = np.vstack((np.ones_like(m), m)).T  # design matrix

    a, e, rsq = multi_regress(y, Z)  # use least square regression to find coeffs, residuals, and R^2

    print(f'a: {a}')
    print(f' e: {e}')
    print(f' rsq: {rsq}')

    Log2 = Z @ a  # compute predicted values

    plt.plot(m, y, 'o', label='data between 35 and 46')
    plt.plot(m, Log2, '-', label='fit 2')

    k_72= int(np.argwhere(t < 72)[-1].item())

    m = np.linspace(-0.5, 1, 9)
    n = np.zeros_like(m)  # array of zeros same size as m

    for i, mm in enumerate(m):  # loop through m and count values in m_data that are greater than or equal to mm
        n[i] = np.count_nonzero(m_data[k_46:k_72] >= mm)  # count occurrences
    print(n)
    y = np.log10(n)
    Z = np.vstack((np.ones_like(m), m)).T  # design matrix

    a, e, rsq = multi_regress(y, Z)  # use least square regression to find coeffs, residuals, and R^2

    print(f'a: {a}')
    print(f' e: {e}')
    print(f' rsq: {rsq}')

    Log3 = Z @ a  # compute predicted values

    plt.plot(m, y, 'o', label='data between 47 and 72')
    plt.plot(m, Log3, '-', label='fit 3')
    plt.xlabel('m')
    plt.ylabel('log10(n)')
    plt.title('log10(n) vs m')
    plt.legend()
    plt.savefig('../figures/log plots.png')
    plt.show()

if __name__ == "__main__":
    main()