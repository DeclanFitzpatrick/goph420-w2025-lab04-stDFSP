
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