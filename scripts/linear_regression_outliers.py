import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import time

import warnings
warnings.filterwarnings('ignore')

from scipy import optimize
import pymc3 as pm
import theano as thno
import theano.tensor as T 

# configure some basic options
sns.set(style="darkgrid", palette="deep")
# pd.set_option('display.notebook_repr_html', True)
plt.rcParams['figure.figsize'] = 12, 8
np.random.seed(0)


#### cut & pasted directly from the fetch_hogg2010test() function
## identical to the original dataset as hardcoded in the Hogg 2010 paper

dfhogg = pd.DataFrame(np.array([[1, 201, 592, 61, 9, -0.84],
                                 [2, 244, 401, 25, 4, 0.31],
                                 [3, 47, 583, 38, 11, 0.64],
                                 [4, 287, 402, 15, 7, -0.27],
                                 [5, 203, 495, 21, 5, -0.33],
                                 [6, 58, 173, 15, 9, 0.67],
                                 [7, 210, 479, 27, 4, -0.02],
                                 [8, 202, 504, 14, 4, -0.05],
                                 [9, 198, 510, 30, 11, -0.84],
                                 [10, 158, 416, 16, 7, -0.69],
                                 [11, 165, 393, 14, 5, 0.30],
                                 [12, 201, 442, 25, 5, -0.46],
                                 [13, 157, 317, 52, 5, -0.03],
                                 [14, 131, 311, 16, 6, 0.50],
                                 [15, 166, 400, 34, 6, 0.73],
                                 [16, 160, 337, 31, 5, -0.52],
                                 [17, 186, 423, 42, 9, 0.90],
                                 [18, 125, 334, 26, 8, 0.40],
                                 [19, 218, 533, 16, 6, -0.78],
                                 [20, 146, 344, 22, 5, -0.56]]),
                   columns=['id','x','y','sigma_y','sigma_x','rho_xy'])


## Read in the data from file straightline.dat
## _r for 'raw'
x_r, y_r, yerr_r, xerr_r = np.loadtxt('straightline.dat', skiprows=2, unpack=True)


## standardize (mean center and divide by 1 sd)
x = (x_r - x_r.mean()) / x_r.std(ddof=1)
y = (y_r - y_r.mean()) / y_r.std(ddof=1)

yerr = yerr_r / y_r.std(ddof=1)
xerr = xerr_r / x_r.std(ddof=1)
dfhoggs = (dfhogg[['x','y']] - dfhogg[['x','y']].mean(0)) / dfhogg[['x','y']].std(0)
dfhoggs['sigma_y'] = dfhogg['sigma_y'] / dfhogg['y'].std(0)
dfhoggs['sigma_x'] = dfhogg['sigma_x'] / dfhogg['x'].std(0)

## create xlims ylims for plotting
xlims = (x.min() - np.ptp(x)/5, x.max() + np.ptp(x)/5)
ylims = (y.min() - np.ptp(y)/5, y.max() + np.ptp(y)/5)

## scatterplot the standardized data
fig = plt.figure('true')
ax = fig.add_subplot(111)
ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_xlim(xlims); ax.set_ylim(ylims)
ax.set_title('Scatterplot of Hogg 2010 dataset after standardization', fontsize=16)
plt.show()



do_ols = False

if do_ols:

    print('OLS analysis...')

    t1 = time.time()
    with pm.Model() as mdl_ols:
        
        ## Define weakly informative Normal priors to give Ridge regression
        b0 = pm.Normal('b', mu=0, sd=100)
        m0 = pm.Normal('m', mu=0, sd=100)
     
        ## Define linear model
        yest = b0 + m0 * x
        
        ## Use y error from dataset, convert into theano variable
        sigma_y = thno.shared(np.asarray(yerr), name='sigma_y')

        ## Define Normal likelihood
        likelihood = pm.Normal('likelihood', mu=yest, sd=sigma_y, observed=y)

    t2 = time.time()
    print ('Created model, took %f seconds' % (t2-t1))


    with mdl_ols:

        ## find MAP using Powell, seems to be more robust
        t1 = time.time()
        start_MAP = pm.find_MAP(fmin=optimize.fmin_powell)
        t2 = time.time()
        print ('Found MAP, took %f seconds' % (t2-t1))

        ## take samples
        t1 = time.time()
        traces_ols = pm.sample(2000, start=start_MAP, step=pm.NUTS(), progressbar=True)
        print()
        t2 = time.time()
        print ('Done sampling, took %f seconds' % (t2-t1))

    pm.summary(traces_ols)
    ## plot the samples and the marginal distributions
    _ = pm.traceplot(traces_ols, figsize=(12,len(traces_ols.varnames)*1.5),
                    lines={k: v['mean'] for k, v in pm.df_summary(traces_ols).iterrows()})
    plt.show()


do_tstudent = False

if do_tstudent:

    print('Robust Student-t analysis...')

    t1 = time.time()
    with pm.Model() as mdl_studentt:
        
        ## Define weakly informative Normal priors to give Ridge regression
        b1 = pm.Normal('b', mu=0, sd=100)
        m1 = pm.Normal('m', mu=0, sd=100)
     
        ## Define linear model
        yest = b1 + m1 * x
        
        ## Use y error from dataset, convert into theano variable
        sigma_y = thno.shared(np.asarray(yerr), name='sigma_y')
        
        ## define prior for Student T degrees of freedom
        nu = pm.DiscreteUniform('nu', lower=1, upper=100)

        ## Define Student T likelihood
        likelihood = pm.StudentT('likelihood', mu=yest, sd=sigma_y, nu=nu, observed=y)

    t2 = time.time()
    print ('Created model, took %f seconds' % (t2-t1))


    with mdl_studentt:

        ## find MAP using Powell, seems to be more robust
        t1 = time.time()
        start_MAP = pm.find_MAP(fmin=optimize.fmin_powell)
        t2 = time.time()
        print ('Found MAP, took %f seconds' % (t2-t1))

        ## two-step sampling to allow Metropolis for nu (which is discrete)
        step1 = pm.NUTS([b1, m1])
        step2 = pm.Metropolis([nu])
        
        ## take samples
        t1 = time.time()
        traces_studentt = pm.sample(2000, start=start_MAP, step=[step1, step2], progressbar=True)
        print()
        t2 = time.time()
        print ('Done sampling, took %f seconds' % (t2-t1))


    _ = pm.traceplot(traces_studentt[-1000:]
                ,figsize=(12,len(traces_studentt.varnames)*1.5)
                ,lines={k: v['mean'] for k, v in pm.df_summary(traces_studentt[-1000:]).iterrows()})
    plt.show()



do_signoise = True

if do_signoise:

    print('Signal/Noise analysis...')

    def logp_signoise(yobs, is_outlier, yest_in, sigma_y_in, yest_out, sigma_y_out):
        '''
        Define custom loglikelihood for inliers vs outliers. 
        '''
        
        # likelihood for inliers
        pdfs_in = T.exp(-(yobs - yest_in + 1e-4)**2 / (2 * sigma_y_in**2)) 
        pdfs_in /= T.sqrt(2 * np.pi * sigma_y_in**2)
        logL_in = T.sum(T.log(pdfs_in) * (1 - is_outlier))

        # likelihood for outliers
        pdfs_out = T.exp(-(yobs - yest_out + 1e-4)**2 / (2 * (sigma_y_in**2 + sigma_y_out**2))) 
        pdfs_out /= T.sqrt(2 * np.pi * (sigma_y_in**2 + sigma_y_out**2))
        logL_out = T.sum(T.log(pdfs_out) * is_outlier)

        return logL_in + logL_out

    t1 = time.time()
    with pm.Model() as mdl_signoise:
        
        ## Define weakly informative Normal priors to give Ridge regression
        b2 = pm.Normal('b', mu=0, sd=100)
        m2 = pm.Normal('m', mu=0, sd=100)
     
        ## Define linear model
        yest_in = b2 + m2 * x

        ## Define weakly informative priors for the mean and variance of outliers
        yest_out = pm.Normal('mu_b', mu=0, sd=100)
        sigma_y_out = pm.HalfNormal('s_b', sd=100)

        ## Define Bernoulli inlier / outlier flags according to a hyperprior 
        ## fraction of outliers, itself constrained to [0,.5] for symmetry
        frac_outliers = pm.Uniform('frac_outliers', lower=0., upper=.5)
        is_outlier = pm.Bernoulli('is_outlier', p=frac_outliers, shape=x.size)  
           
        ## Extract observed y and sigma_y from dataset, encode as theano objects
        yobs = thno.shared(np.asarray(y), name='yobs')
        sigma_y_in = thno.shared(np.asarray(yerr), name='sigma_y_in')
            
        ## Use custom likelihood using DensityDist
        likelihood = pm.DensityDist('likelihood', logp_signoise,
                            observed={'yobs':yobs, 'is_outlier':is_outlier,
                                      'yest_in':yest_in, 'sigma_y_in':sigma_y_in,
                                      'yest_out':yest_out, 'sigma_y_out':sigma_y_out})

    t2 = time.time()
    print ('Created model, took %f seconds' % (t2-t1))


    with mdl_signoise:

        ## two-step sampling to create Bernoulli inlier/outlier flags
        step1 = pm.NUTS([frac_outliers, yest_out, sigma_y_out, b2, m2])
        step2 = pm.BinaryMetropolis([is_outlier], tune_interval=100)

        ## find MAP using Powell, seems to be more robust
        t1 = time.time()
        start_MAP = pm.find_MAP(fmin=optimize.fmin_powell)
        t2 = time.time()
        print ('Found MAP, took %f seconds' % (t2-t1))

        ## take samples
        t1 = time.time()
        traces_signoise = pm.sample(2000, start=start_MAP, step=[step1,step2], progressbar=True)
        print()
        t2 = time.time()
        print ('Done sampling, took %f seconds' % (t2-t1))


    _ = pm.traceplot(traces_signoise[-1000:], figsize=(12,len(traces_signoise.varnames)*1.5),
                lines={k: v['mean'] for k, v in pm.df_summary(traces_signoise[-1000:]).iterrows()})
    plt.show()


    ## fancy plot stuff
    outlier_melt = pd.melt(pd.DataFrame(traces_signoise['is_outlier', -1000:],
                                        columns=['[{}]'.format(int(d)) for d in dfhoggs.index]),
                          var_name='datapoint_id', value_name='is_outlier')
    ax0 = sns.pointplot(y='datapoint_id', x='is_outlier', data=outlier_melt,
                       kind='point', join=False, ci=None, size=4, aspect=2)

    _ = ax0.vlines([0,1], 0, 19, ['b','r'], '--')

    _ = ax0.set_xlim((-0.1,1.1))
    _ = ax0.set_xticks(np.arange(0, 1.1, 0.1))
    _ = ax0.set_xticklabels(['{:.0%}'.format(t) for t in np.arange(0,1.1,0.1)])

    _ = ax0.yaxis.grid(True, linestyle='-', which='major', color='w', alpha=0.4)
    _ = ax0.set_title('Prop. of the trace where datapoint is an outlier')
    _ = ax0.set_xlabel('Prop. of the trace where is_outlier == 1')
    plt.show()


    cutoff = 5
    dfhoggs['outlier'] = np.percentile(traces_signoise[-1000:]['is_outlier'],cutoff, axis=0)
    dfhoggs['outlier'].value_counts()



def plot_posterior_predictive(trace, color=None, fig=None):

    ## Posterior predictive plots for OLS, StudentT, SignalNoise
    if fig is None:
        try:
            g = sns.FacetGrid(dfhoggs, size=8, hue='outlier', hue_order=[True,False],
                              palette='Set1', legend_out=False)
        except KeyError:
            g = sns.FacetGrid(dfhoggs, size=8, palette='Set1', legend_out=False)        
    else:
        g = fig

    lm = lambda x, samp: samp['b'] + samp['m'] * x

    pm.glm.plot_posterior_predictive(trace[-1000:], lm=lm, 
            eval=np.linspace(-3, 3, 10), samples=200, color=color, alpha=.2)

    # pm.glm.plot_posterior_predictive(traces_ols[-1000:], lm=lm, 
    #         eval=np.linspace(-3, 3, 10), samples=200, color='#22CC00', alpha=.2)

    # pm.glm.plot_posterior_predictive(traces_studentt[-1000:], lm=lm,
    #         eval=np.linspace(-3, 3, 10), samples=200, color='#FFA500', alpha=.5)

    # pm.glm.plot_posterior_predictive(traces_signoise[-1000:], lm=lm,
    #         eval=np.linspace(-3, 3, 10), samples=200, color='#357EC7', alpha=.3)

    _ = g.map(plt.errorbar, 'x', 'y', 'sigma_y', 'sigma_x', marker="o", ls='').add_legend()

    # _ = g.axes[0][0].annotate('OLS Fit: Green\nStudent-T Fit: Orange\nSignal Vs Noise Fit: Blue',
    #                           size='x-large', xy=(1,0), xycoords='axes fraction',
    #                           xytext=(-160,10), textcoords='offset points')
    _ = g.axes[0][0].set_ylim(ylims)
    _ = g.axes[0][0].set_xlim(xlims)
    plt.show()
    return g


# fig = plot_posterior_predictive(traces_ols, color='#22CC00')
# fig = plot_posterior_predictive(traces_studentt, color='#FFA500', fig=fig)
# fig = plot_posterior_predictive(traces_signoise, color='#357EC7', fig=fig)

fig = plot_posterior_predictive(traces_signoise, color='#357EC7')