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



## Lecture 1
	- Start with the example from d'Agostini on the meaning of frequentist and Bayesian probabilities

	- Use a simple example to show all the problems with frequentist p-values and confidence intervals
		. http://www.indiana.edu/~kruschke/articles/Kruschke2013JEPG.pdf
		. they don't actually depend on the data we have at hand
		. they are subjective too, in the sense that they depend on termination and other researcher degrees of freedom

	- State the two laws of probability and derive Bayes' theorem from them
		. explain in extreme detail what every term is

	- The linear regression example
		. we should start by building a pgm for the problem
		. then, derive the posterior distribution for the parameters m and b
		. sample with MCMC

	- The beta-binomial conjugacy 
		. what does it mean to be a conjugate prior?

	- Build a hierarchical model
		. maybe use the HARPS offset as an example?



This work is licensed under a [Attribution-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-sa/4.0/) license.
