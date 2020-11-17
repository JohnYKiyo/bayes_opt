from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
import jax.scipy as scp
from jax import jit


@jit
def UCB(mu, sigma, kappa=5., **kwargs):
    return mu + kappa * sigma


@jit
def LCB(mu, sigma, kappa=5., **kwargs):
    return -mu + kappa * sigma


@jit
def scheduledLCB(mu, sigma, kappa=5., **kwargs):
    it = kwargs.get('it')
    schedule = np.sqrt(np.log(it + 1) / (it + 1))
    return -mu + kappa * schedule * sigma


@jit
def scheduledUCB(mu, sigma, kappa=5., **kwargs):
    it = kwargs.get('it')
    schedule = np.sqrt(np.log(it + 1) / (it + 1))
    return mu + kappa * schedule * sigma


@jit
def MinPI(mu, sigma, xi=0.01, **kwargs):
    vmin = kwargs.get('vmin')
    Z =(vmin - mu - xi)/sigma 
    return scp.stats.norm.cdf(Z)


@jit
def MaxPI(mu, sigma, xi=0.01, **kwargs):
    vmax = kwargs.get('vmax')
    Z = (mu - vmax - xi)/sigma
    return scp.stats.norm.cdf(Z)


@jit
def MinEI(mu, sigma, **kwargs):
    vmin = kwargs.get('vmin')
    Z=(vmin - mu) / sigma
    return np.where(sigma==0, 0, sigma * (Z * scp.stats.norm.cdf(Z) + scp.stats.norm.pdf(Z)))


@jit
def MaxEI(mu, sigma, **kwargs):
    vmax = kwargs.get('vmax')
    Z=(mu - vmax) / sigma
    return np.where(sigma==0, 0, sigma * (Z * scp.stats.norm.cdf(Z) + scp.stats.norm.pdf(Z)))
