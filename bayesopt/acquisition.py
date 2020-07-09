from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import vmap,jit

@jit
def UCB(mu,sigma,kappa=5.):
    return mu+kappa*sigma

def PI(*args,**kwargs):
    raise NotImplementedError

def EI(*args,**kwargs):
    raise NotImplementedError