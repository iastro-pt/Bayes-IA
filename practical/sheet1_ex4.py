import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ...

# number of iterations to run MCMC


# set p as the posterior distribution (the distribution we want to sample from)
# use distributions from Scipy


# initial condition
x = []


# number of accepted steps
n_accept = 0


# iterate
for step in :
	# propose a step
	d = np.random.

	# calculate the ratio of posterior probabilities
	ratio =

	if ratio > 1.:
		# accept the new step
		# increase n_accept
	else:
		# accept with probability = ratio


# acceptance ratio
acceptance = 
