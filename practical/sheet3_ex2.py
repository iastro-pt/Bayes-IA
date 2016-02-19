"""
Exercise 2 of problem sheet 'Linear Regression with outliers'
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano

## Read in the data from file straightline.dat
x_r, y_r, yerr_r, xerr_r = np.loadtxt('straightline.dat', skiprows=2, unpack=True)


## standardize (mean center and divide by 1 sd)
x = (x_r - x_r.mean()) / x_r.std(ddof=1)
y = (y_r - y_r.mean()) / y_r.std(ddof=1)

yerr = yerr_r / y_r.std(ddof=1)


## scatterplot of the standardized data?

# plt.figure()
# plt.errorbar(x, y, yerr, fmt='o')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


## set overall parameters for the MCMC
niter = 2000
burnin = 1000


## use pm.Model() in a with statement as in
with pm.Model() as model1:
        
    ## Define weakly informative Normal priors for b and m
    b = pm.Normal('b', mu=0, sd=100)
    m = pm.Normal('m', mu=0, sd=100)

	## Define linear model
    ymodel = m*x + b 

    ## Use the y error (but convert it to a theano.shared variable)
    sigma_y = theano.shared(yerr, name='sigma_y')


    ## Define the likelihood
    likelihood = pm.Normal('likelihood', mu=ymodel, sd=sigma_y, observed=y)


## use the model you just created in a with statement
with model1:

    ## find MAP value of the parameters
    start_MAP = pm.find_MAP()

    ## sample using the MCMC algorithm
    ## use the pm.NUTS() algorithm as the step
    samples = pm.sample(niter, start=start_MAP, step=pm.NUTS(), progressbar=True)

## print a summary of the results
pm.summary(samples[burnin:])



## plot the samples and the marginal distributions
## using the built-in PyMC3 functions
pm.traceplot(samples[burnin:])
plt.show()


# the following two definitions of the function 'lm' are equivalent
def lm(x, samp):
    return samp['b'] + samp['m']*x

lm = lambda x, samp: samp['b'] + samp['m']*x


## plot the posterior predictive samples
plt.figure()
plt.errorbar(x, y, yerr, fmt='o')
pm.glm.plot_posterior_predictive(samples, lm=lm, samples=100, alpha=0.3,
                                 eval=np.linspace(min(x), max(x)))
plt.show()


