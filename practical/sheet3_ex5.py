"""
Exercise 2 of problem sheet 'Linear Regression with outliers'
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pymc3 as pm
import theano
import theano.tensor as T 

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

    def loglike(yobs, qi, ymodel_in, sigma_y_in, mub, sb):
        '''
        Define custom loglikelihood for inliers and outliers
        This is Eq 13 of the problem sheet but note the typo: in Eq 13
        where you read [1-qi] it should read qi
        and where you read qi it should read [1-qi].
        See our definition of the qi below.
        '''
        # variances
        Vi = sigma_y_in ** 2
        Vb = sb ** 2

        logL_in = -0.5 * T.sum(qi * (T.log(2 * np.pi * Vi)
                                      + (yobs - ymodel_in) ** 2 / Vi))

        logL_out = -0.5 * T.sum((1 - qi) * (T.log(2 * np.pi * (Vi + Vb))
                                             + (yobs - mub) ** 2 / (Vi + Vb)))

        return logL_out + logL_in


    ## Define weakly informative Normal priors for b and m
    b = pm.Normal('b', mu=0, sd=100)
    m = pm.Normal('m', mu=0, sd=100)

    ## Define linear model
    ymodel = m*x + b 


    ## Define weakly informative priors for the mean and variance of outliers
    mub = pm.Normal('mu_b', mu=0, sd=100)
    sb = pm.HalfNormal('s_b', sd=100)

    # fraction of outliers
    Pb = pm.Uniform('Pb', lower=0., upper=1.)

    # one qi per data point#
    # qi is 1 if the ith data point is good, and 0 if the ith data point is bad
    qi = pm.Bernoulli('qi', p=1-Pb, shape=x.size)  

    ## Use the y error (but convert it to a theano.shared variable)
    sigma_y = theano.shared(yerr, name='sigma_y')
    ## Also need to convert y to a theano variable
    yobs = theano.shared(y, name='yobs')


    ## Group all the arguments needed for the likelihood function loglike
    args = {'yobs':yobs, 'qi':qi,
            'ymodel_in':ymodel, 'sigma_y_in':sigma_y,
            'mub':mub, 'sb':sb}

    ## Define the likelihood
    # the pm.DensityDist allows us to use the loglike function 
    # we defined above as the likelihood
    likelihood = pm.DensityDist('likelihood', loglike, observed=args)


## use the model you just created in a with statement
with model1:

    ## find MAP value of the parameters
    start_MAP = pm.find_MAP(fmin=optimize.fmin_powell)

    ## sample using the MCMC algorithm
    # the qi parameters need a special step because they can only be either 0 or 1
    step1 = pm.NUTS([Pb, mub, sb, b, m])
    step2 = pm.BinaryMetropolis([qi], tune_interval=100)
    step = [step1, step2]

    samples = pm.sample(niter, start=start_MAP, step=[step1, step2], progressbar=True)


## Declare a point as an outlier if its qi is 0 in more than 99% of the MCMC samples
cutoff = 1
outlier = np.percentile(1 - samples[burnin:]['qi'], cutoff, axis=0)
outlier = outlier.astype(bool)
# the variable 'outlier' is an array of size N with True for outlier points and False for inlier points
# the points that are identified as outlier can change from run to run, especially if niter is small

## print a summary of the results
pm.summary(samples[burnin:])


## plot the samples and the marginal distributions
## using the built-in PyMC3 functions
pm.traceplot(samples[burnin:])
plt.show()


## in the previous plots and results you will also see
## the parameters s_b_log and Pb_interval
## these are created automatically by PyMC3 
## but we don't need to worry about them


# the following two definitions of the function 'lm' are equivalent
def lm(x, samp):
    return samp['b'] + samp['m']*x

lm = lambda x, samp: samp['b'] + samp['m']*x


## plot the posterior predictive samples
plt.figure()

plt.errorbar(x, y, yerr, fmt='o') # plot all the points
plt.errorbar(x[outlier], y[outlier], yerr[outlier], fmt='o', color='r') # plot the outliers in red

pm.glm.plot_posterior_predictive(samples, lm=lm, samples=100, alpha=0.3,
                                 eval=np.linspace(min(x), max(x)))
plt.show()


