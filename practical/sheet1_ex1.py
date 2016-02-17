import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# number of iterations to run MCMC
niter = 1000

# set p as the posterior distribution (the distribution we want to sample from)
def p(x):
	return norm(loc=0, scale=1.).pdf(x)

# initial condition
x = [0.1]


# number of accepted steps
n_accept = 0

# iterate
for step in range(niter):
	# propose a step
	d = 1*np.random.randn()

	# candidate
	xnew = x[-1] + d

	# calculate the ratio of posterior probabilities
	ratio = p(xnew) / p(x[-1])

	if ratio > 1.:
		# accept the new step
		x.append(xnew)
		n_accept += 1
	else:
		# accept with probability = ratio
		u = np.random.uniform()
		if ratio > u:
			x.append(xnew) # accept the new step
			n_accept += 1
		else:
			x.append(x[-1])
 

print 'Acceptance fraction', n_accept/float(niter)


t = np.linspace(-6, 6, 1000)
plt.figure()
plt.subplot(211)
plt.plot(t, p(t), 'k-', lw=3, label='target distribution')
plt.hist(x, bins=50, normed=True, alpha=0.7, label='samples')
plt.axvline(np.mean(x), lw=4, ls='--', color='r', label='mean of samples')
plt.xlim([-5, 5])
# plt.legend()
plt.subplot(212)
plt.plot(x)
plt.xlabel('iteration')
plt.ylabel(r'sampled $x$')
plt.xlim([-0.05*niter, 1.05*niter])
plt.show()