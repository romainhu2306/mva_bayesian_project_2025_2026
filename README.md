# Bayesian Machine Learning Project: When is Generalized Bayes Bayesian ? A Decision Theoretic Characterization of Loss-Based Updating.

By Romain Hû and Perann Nedjar

This is the repository attached to our work on the paper by McAlinn and Takanashi (2026). It contains our report and two notebooks illustrating the interest and properties of Gibbs learning on simple toy models.


## misspecification_illustration.ipynb

This notebook showcases the problem of estimating the mean of a mixture of centered gaussian distributions with outliers. It compares the performance and posterior concentration of 3 models with normal priors:

- A standard Bayesian model with gaussian likelihood: $p(\theta|y) = p(y|\theta)\pi(\theta)$ ;
- A tempered model of the previous version with small confidence parameter $\eta$: $p(\theta|y) = p(y|\theta)^\eta \pi(\theta)$
- A Gibbs model using the Huber loss: $p(\theta|y) = exp(-\eta l(\theta, y)) \pi(\theta)$

Results showcase the benefits of choosing an adapted loss with Gibbs learning. It also shows how tempered Bayesian models can improve accuracy at the cost of concentration, illustrating the lack of confidence interpretation.


## linear_regression_example.ipynb

This notebook showcases certain properties of Gibbs learning for simple prediction tasks.

It illustrates the influence of the choice of loss and parameter $eta$ on the posterior concentration. Posteriors are computed using the MH sampling algorithm.

It also empirically shows that data only transformations of the loss preserve the Gibbs posterior but affect the normalization $Z$.

Finally, randomized predictors are used on a misspecified setting, where linear data is fitted with an exponential model $y = exp(\theta x)$, showing how randomized predictions can acquire properties that go beyond the original model class.


## GeneralBayesFunctions.py

Contains utils functions for the linear regression example, including losses and sampling functions.