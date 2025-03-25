import numpy as np

def multi_regress(y, Z):
    """Perform multiple linear regression.
    Parameters
    ----------
    y : array_like, shape = (n,) or (n,1)
    The vector of dependent variable data
    Z : array_like, shape = (n,m)
    The matrix of independent variable data
    Returns
    -------
    numpy.ndarray, shape = (m,) or (m,1)
    The vector of model coefficients
    numpy.ndarray, shape = (n,) or (n,1)
    The vector of residuals
    float
    The coefficient of determination, r^2
    """
    
    y = np.ndarray.flatten(y)
    z = np.ndarray.flatten(Z)   

    a = np.linalg.inv(np.transpose(z)*z)*(np.transpose(z)*y)  
    print(f'a = {a}')  
    e = y - z*a
    print(f'e = {e}')