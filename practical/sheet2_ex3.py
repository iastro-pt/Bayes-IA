import numpy as np
import matplotlib.pyplot as plt

## Read in the data from file straightline.dat
# only data points 5 through 20
x, y, yerr, xerr = np.loadtxt('straightline.dat', skiprows=6, unpack=True)

print 'Read %d data points' % x.size


## least-squares solution
# build the matrix A (you can use the np.vstack function)
A = np.vstack((np.ones_like(x), x, x**2)).T

# build the matrix C
C = np.diag(yerr**2)

# calculate the covariance matrix [A^T C^-1 A]^-1
# (use the linear algebra functions in np.linalg)
cov = np.dot(A.T, np.linalg.solve(C, A))
cov2 = np.linalg.inv(cov)

# calculate the X matrix
X = np.dot(cov2, np.dot(A.T, np.linalg.solve(C,y) ))

# extract from X the parameters m and b
b, m, q = X 

print 'b=', b, '+-', np.sqrt(cov2[0,0]) 
print 'm=', m, '+-', np.sqrt(cov2[1,1])
print 'q=', q, '+-', np.sqrt(cov2[2,2])

# plot the data (with errorbars) and the best-fit line
plt.figure()
plt.errorbar(x, y, yerr, fmt='o')

xx = np.linspace(min(x), max(x))
plt.plot(xx, q*xx**2 + m*xx + b, '-')

plt.show()

