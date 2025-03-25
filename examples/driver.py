import numpy as np

from src.lab_04.regression import (multi_regress)

def main():
    data = np.loadtxt("M_data1.txt")
    y = data[:,0] # enter dependent variable data (or load from a file)
    Z = data[:,1] # enter independent variable data (or load from a file)
    a, e, rsq = multi_regress(y, Z)



if __name__ == "__main__":
    main()