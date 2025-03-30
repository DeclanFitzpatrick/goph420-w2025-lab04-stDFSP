import numpy as np


def multi_regress(y, Z):
    """
    Perform multiple linear regression.

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
    The vector of residuals, float
    The coefficient of determination, r^2
    """

    y = np.array(y)
    z = np.array(Z)

    a = np.linalg.solve(z.T @ z, z.T @ y)  # compute regression coeff
    e = y - z @ a  # compute residuals

    sr = e.T @ e  # compute sum of squared residuals

    ey = y-np.mean(y)
    st = ey.T @ ey  # compute total sum of squares

    rsq = 1 - sr/st  # compute coeff of determination

    return a, e, rsq