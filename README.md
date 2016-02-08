Data don't speak for themselves

Description
===========

Bayesian statistics is rising in popularity in the astrophysical literature.
It is no longer a debate: "work in Bayesian statistics now focuses on applications, 
computations, and models. Philosophical debates [...] are fading to the background" 
(Bayesian Data Analysis, Gelman et al.).
This is happening for two main reasons: faster computers and more complex models. 
In order to keep up, it is important to understand the fundamentals of Bayesian statistics,
but it is as important to know how to deal with data analysis applications.

In this course I want to provide a brief introduction to advanced concepts in Bayesian statistics.
Emphasis will be on "intuition" and "computation". No coin tossing, only real applications that
relate to our day-to-day problems. 



Plan for the course
===================


The idea is to present some statistical results in an intuitive manner
and then turn to computational methods.
By the end of the course, the students should be able to understand:

THEORY

 - what are the main differences between frequentist and Bayesian statistics
 	. problems with p-values, confidence intervals and null hypothesis testing
 - the basic rules of probability theory
 - how to assign probability distributions
 	. the role of priors
 	. the likelihood
 - the simplest models in Bayesian statistics: 
	. linear regression, 
	. beta-binomial model, 
	. hierarchical models

PRACTICE

 - what is an MCMC, how it works and why it works; shortcomings (and alternatives)
 - code an MCMC from scratch to sample a distribution
 - for the (weighted) linear regression model
	. build the probabilistic graphical model and derive the posterior distribution
	. use our MCMC to sample the posterior for slope and intercept
	. solve the problem with a "black box" MCMC sampler in Python
 - calculate the evidence integral to do model comparison



This work is licensed under a [Attribution-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-sa/4.0/) license.
