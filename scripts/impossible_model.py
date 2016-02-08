"""
A two-parameter model that does not fit a dataset with two observations
"""

import numpy as np
from lmfit import  Model
import matplotlib.pyplot as plt

x = np.linspace(0.1, 0.5, 2)
y = 2.1*x + 3

def line(x, m, b):
    return m*x + b

gmod = Model(line)

# this makes the model impossible
gmod.set_param_hint('m', min=-10.0, max=0.)

result = gmod.fit(y, x=x, m=1, b=0)

print(result.fit_report())

plt.plot(x, y, 'bo')
plt.plot(x, result.init_fit, 'k--')
plt.plot(x, result.best_fit, 'r-')
plt.xlim([0, 0.6])
plt.show()