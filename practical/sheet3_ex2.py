"""
Exercise 2 of problem sheet 'Linear Regression with outliers'
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

## Read in the data from file straightline.dat


## standardize the data (mean center and divide by 1 sd)


## scatterplot of the standardized data?


## set overall parameters for the MCMC
niter = 2000
burnin = 1000


## use pm.Model() in a with statement as in
# with pm.Model() as model1:
        
    ## Define weakly informative Normal priors for b and m
    b = 
    m = 

	## Define linear model


    ## Use the y error (but convert it to a theano.shared variable)
    # sigma_y = thno.shared(####, name='sigma_y')


    ## Define the likelihood
    likelihood = 


## use the model you just created in a with statement
# with model1:

    ## find MAP value of the parameters
    start_MAP = 

    ## sample using the MCMC algorithm
    ## use the pm.NUTS() algorithm as the step
    samples = 

    ## print a summary of the results




## plot the samples and the marginal distributions
## using the built-in PyMC3 functions


## plot the posterior predictive samples