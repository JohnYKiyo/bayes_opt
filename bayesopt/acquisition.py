from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
import jax.scipy as scp
from jax import jit

@jit
def UCB(mu,sigma,kappa=5.):
    return mu + kappa*sigma

@jit
def LCB(mu,sigma,kappa=5.,**kwargs):
    return -mu+kappa*sigma

@jit
def scheduledLCB(mu,sigma,kappa=5.,**kwargs):
    it = kwargs.get('it')
    schedule = np.sqrt(np.log(it+1) / (it+1))
    return -mu+kappa*schedule*sigma

@jit
def PI(mu,sigma,xi=0.01,**kwargs):
    vmin = kwargs.get('vmin')
    return scp.stats.norm.cdf((vmin-mu-xi)/sigma)

@jit
def EI(mu,sigma,**kwargs):
    vmin = kwargs.get('vmin')
    Z=(vmin-mu)/sigma
    return np.where(sigma==0,0,(vmin-mu)*scp.stats.norm.cdf(Z)+sigma*scp.stats.norm.pdf(Z))

def GunbelUCB(mu,sigma,**kwargs):
    raise NotImplementedError