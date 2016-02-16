import numpy as np

## Read in the data from file straightline.dat
x, y, yerr, xerr = 

print 'Read %d data points' % x.size


## least-squares solution
# build the matrix A (you can use the np.vstack function)
A = 

# build the matrix C
C = 

# calculate the covariance matrix [A^T C^-1 A]^-1
# (use the linear algebra functions in np.linalg)
cov = 

# calculate the X matrix
X = 

# extract from X the parameters m and b
b, m = 





# plot the data (with errorbars) and the best-fit line