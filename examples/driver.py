import numpy as np
import matplotlib.pyplot as plt
from lab_04.regression import (multi_regress)


def main():
    data = np.loadtxt("../data/M_data1.txt")

    t = data[:, 0]  # dependent data
    m_data = data[:, 1]  # independent data

    data_segs = [(0, 34), (35, 46), (47, 72), (73, 96)]
    m = np.linspace(0, 1, 15)
    n = np.zeros_like(m)  # array of zeros same size as m

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    for i, (first, last) in enumerate(data_segs):
        if first == 0:
            k_first = 0
        else:
            k_first = int(np.argwhere(t < first)[-1].item())  # to extract scalar value and not return array
        k_last = int(np.argwhere(t < last)[-1].item())

        for j, mm in enumerate(m):  # loop through m and count values in m_data that are greater than or equal to mm
            n[j] = np.count_nonzero(m_data[k_first:k_last] >= mm)  # count occurrences

        y = np.log10(n)
        Z = np.vstack((np.ones_like(m), m)).T  # design matrix

        a, e, rsq = multi_regress(y, Z)  # use least square regression to find coeffs, residuals, and R^2

        # for report and debugging
        print(f"{first}–{last} hours:")
        print(f'intercept (a): {a[0]}')
        print(f'slope (b): {a[1]}')
        print(f'e: {e}')
        print(f'e average: {np.mean(e)}')
        print(f'R²: {rsq}')

        log_n = Z @ a  # compute predicted values

        # plotting...
        axes[i].plot(m, log_n, 'b-', linewidth=2, label='Regression Line')
        axes[i].plot(m, y, 'ro', label=f'$y = {a[0]:.2f} + ({a[1]:.2f})m$', markersize=6)
        axes[i].set_xlabel('Magnitude', fontsize=10)
        axes[i].set_ylabel(r'$\log_{10}(n)$', fontsize=10)
        axes[i].set_title(f'{first} to {last} hours', fontsize=12, fontweight='bold')
        axes[i].legend(loc='upper right', fontsize=8, frameon=True, facecolor='white', edgecolor='black')
        axes[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('../figures/log_plots.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
