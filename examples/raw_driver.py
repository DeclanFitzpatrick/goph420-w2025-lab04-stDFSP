import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.loadtxt("../data/M_data1.txt")

    t = data[:,0]  # dependent data
    m_data = data[:,1]  # independent data

    k_120 = np.argwhere(t < 120)[-1]  # find indices when t is less than 120 and select last index
    print(f'time (120): {k_120}')
    print(f'time at k_120: {t[k_120]}')

    k_120 = int(k_120)
    t = t[0:k_120]  # slice t to only include values less than 120
    print(t[0:k_120])

    mk_120 = np.argwhere(m_data < 120)[-1]  # find indices when t is less than 120 and select last index
    print(f'm data (120): {mk_120}')
    print(f'm data at mk_120: {m_data[mk_120]}')

    m_data = m_data[0:k_120]  # slice m_data to only include values less than 120
    print(m_data[0:k_120])

    plt.figure(figsize=(12,6))
    plt.plot(t, m_data, 'go', markersize=2)
    x_list = [34, 46, 72, 96]  # via testing
    for x in x_list:
        plt.axvline(x=x, color='r',linestyle='--')
        plt.text(x + 0.5, max(m_data), f"{x} hours", color='r', fontsize=10)
    plt.xlabel('time')
    plt.ylabel('earthquake magnitude')
    plt.title('Earthquake Magnitude as a function of Time')
    plt.savefig('../figures/magnitude_vs_time.png')
    plt.show()


if __name__ == "__main__":
    main()