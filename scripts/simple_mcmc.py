import numpy as np
from pylab import *
from scipy.stats import norm
from scipy.stats import t as tstudent
from scipy.stats import beta
import time

# number of iterations to run MCMC
niter = 5000

# posterior distribution (the distribution we want to sample from)
p = lambda x: 4*norm.pdf(x)
# p = lambda x:tstudent.pdf(x, 3)
# p = lambda x:beta.pdf(x, 2, 5)

# initial condition
x = [0.5]

t1 = time.time()
for step in range(niter):
	# propose a step
	d = np.random.randn()
	
	# calculate the ratio of posterior probabilities
	ratio = p(x[-1]+d) / p(x[-1])

	if ratio > 1.:
		x.append(x[-1]+d)	# accept the new step
	else:
		u = np.random.uniform()
		if ratio > u:
			x.append(x[-1]+d)	# accept the new step
		else:
			x.append(x[-1])		# reject the new step (stay where we are)
t2 = time.time()

print 'The MCMC took %f seconds' % (t2-t1)


t = np.linspace(-6, 6, 1000)
figure()
subplot(211)
plot(t, p(t) / (4*norm.cdf(1)), 'k-', lw=3, label='target distribution')
hist(x, bins=50, normed=True, alpha=0.7, label='samples')
axvline(np.mean(x), lw=4, ls='--', color='r', label='mean of samples')
xlim([-5, 5])
legend()
subplot(212)
plot(x)
xlabel('iteration')
ylabel(r'sampled $x$')
xlim([-0.05*niter, 1.05*niter])
show()

