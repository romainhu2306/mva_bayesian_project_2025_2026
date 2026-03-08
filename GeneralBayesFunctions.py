import numpy as np
import pymc as pm 
import pymc.math as pm_math
import arviz as az

def cum_loss_L1(x, y, theta):
    return pm_math.sum(pm_math.abs(y - x*theta))

def cum_loss_L2(x, y, theta):
    return pm_math.sum((y - x*theta)**2)

def MHSample(n_sample : int, X : np.array , Y : np.array, tau : float, eta : float , loss):
    
    with pm.Model() as model:
        theta = pm.Normal("theta", mu=0, sigma=tau)
        pm.Potential("loss", -eta * loss(X, Y, theta))
        
        step = pm.Metropolis()
        trace = pm.sample(n_sample, tune=1000, step=step, cores=1, chains=2, progressbar=True)

    rhat = az.rhat(trace, var_names=["theta"])
    ess = az.ess(trace, var_names=["theta"])
    print("R-hat:", rhat["theta"].values)
    print("Effective sample size (ESS):", ess["theta"].values)
    
    return trace