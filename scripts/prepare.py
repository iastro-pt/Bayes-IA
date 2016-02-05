import sys

print (sys.version)
print ('')

def print_error():
	print ('\nThere are some packages missing. Please install them.')


try:
	import numpy
	import scipy
	import matplotlib
except ImportError:
	print( 'You need to have numpy, scipy and matplotlib installed.')
	print_error()
	sys.exit(1)

try:
	import emcee
except ImportError:
	print( "You need to install 'emcee'. Run")
	print( '  pip install emcee')
	print( "or visit http://dan.iel.fm/emcee/current/user/install/")
	print_error()
	sys.exit(1)


print( 'numpy', numpy.__version__)
print( 'Scipy', scipy.__version__)
print( 'matplotlib', matplotlib.__version__)
print('')

try:
	import pymc3 as pm
	try:
		import theano
		import theano.tensor
	except ImportError:
		print( 'ERROR: Something went wrong with Theano!')
		sys.exit(0)
except ImportError:
	print( "You need to install 'pymc3'. Run")
	print( '  pip install --process-dependency-links git+https://github.com/pymc-devs/pymc3')
	print( "or visit http://pymc-devs.github.io/pymc3/getting_started/#installation")
	print_error()
	sys.exit(1)		


print( 'theano', theano.__version__)
print('')

try:
	import pandas as pd
except ImportError:
	print( "Though not strictly necessary, you could try installing 'pandas'. Run")
	print( "  pip install pandas")
	print( "or go to https://github.com/pydata/pandas for more information")

try:
	import seaborns as sns
except ImportError:
	print( "Though not strictly necessary, you could try installing 'seaborn'. Run")
	print( "  pip install seaborn")
	print( "or go to http://stanford.edu/~mwaskom/software/seaborn/installing.html for more information")



print( "\nAll packages working! You're good to go")