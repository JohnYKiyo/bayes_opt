from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import jax.scipy as scp
from jax import jit


@jit
def UCB(mu, sigma, kappa=5., **kwargs):
    """Acquisition function using the upper confidence bound.
    Args:
        mu (float): Mean of functions estimated by Gaussian process.
        sigma (float): Function variance estimated by the Gaussian process.
        kappa (float, optional): The coefficient of the search term. The higher the value, the stronger the search tendency. kappa >=0. Defaults to 5..

    Returns:
        float: Acquisition function value.
    """
    return mu + kappa * sigma


@jit
def LCB(mu, sigma, kappa=5., **kwargs):
    """Acquisition function using the lower confidence bound.
    Args:
        mu (float): Mean of functions estimated by Gaussian process.
        sigma (float): Function variance estimated by the Gaussian process.
        kappa (float, optional): The coefficient of the search term. The higher the value, the stronger the search tendency. kappa >=0. Defaults to 5..

    Returns:
        float: Acquisition function value.
    """
    return -mu + kappa * sigma


@jit
def scheduledLCB(mu, sigma, kappa=5., **kwargs):
    """A scheduled LCB acquisition function that strengthens the search term as the evaluation progresses.
    Args:
        mu (float): Mean of functions estimated by Gaussian process.
        sigma (float): Function variance estimated by the Gaussian process.
        kappa (float, optional): The coefficient of the search term. The higher the value, the stronger exploration. kappa >=0. Defaults to 5..

    Returns:
        float: Acquisition function value.
    """
    it = kwargs.get('it')
    schedule = np.sqrt(np.log(it + 1) / (it + 1))
    return -mu + kappa * schedule * sigma


@jit
def scheduledUCB(mu, sigma, kappa=5., **kwargs):
    """A scheduled UCB acquisition function that strengthens the search term as the evaluation progresses.
    Args:
        mu (float): Mean of functions estimated by Gaussian process.
        sigma (float): Function variance estimated by the Gaussian process.
        kappa (float, optional): The coefficient of the search term. The higher the value, the stronger exploration. kappa >=0. Defaults to 5..

    Returns:
        float: Acquisition function value.
    """
    it = kwargs.get('it')
    schedule = np.sqrt(np.log(it + 1) / (it + 1))
    return mu + kappa * schedule * sigma


@jit
def MinPI(mu, sigma, xi=0.01, **kwargs):
    """Probability of Improvement acquisition function.
    Args:
        mu (float): Mean of functions estimated by Gaussian process.
        sigma (float): Function variance estimated by the Gaussian process.
        xi (float, optional): The trade-off parameter. The higher the value, the stronger exploration. xi >=0. Defaults to 0.01.

    Returns:
        float: Acquisition function value.
    """
    vmin = kwargs.get('vmin')
    Z = (vmin - mu - xi) / sigma
    return scp.stats.norm.cdf(Z)


@jit
def MaxPI(mu, sigma, xi=0.01, **kwargs):
    """Probability of Improvement acquisition function.
    Args:
        mu (float): Mean of functions estimated by Gaussian process.
        sigma (float): Function variance estimated by the Gaussian process.
        xi (float, optional): The trade-off parameter. The higher the value, the stronger exploration. xi >=0. Defaults to 0.01.

    Returns:
        float: Acquisition function value.
    """
    vmax = kwargs.get('vmax')
    Z = (mu - vmax - xi) / sigma
    return scp.stats.norm.cdf(Z)


@jit
def MinEI(mu, sigma, **kwargs):
    """Expected Improvement acquisition function.
    Args:
        mu (float): Mean of functions estimated by Gaussian process.
        sigma (float): Function variance estimated by the Gaussian process.

    Returns:
        float: Acquisition function value.
    """
    vmin = kwargs.get('vmin')
    Z = (vmin - mu) / sigma
    return np.where(sigma == 0, 0, sigma * (Z * scp.stats.norm.cdf(Z) + scp.stats.norm.pdf(Z)))


@jit
def MaxEI(mu, sigma, **kwargs):
    """Expected Improvement acquisition function.
    Args:
        mu (float): Mean of functions estimated by Gaussian process.
        sigma (float): Function variance estimated by the Gaussian process.

    Returns:
        float: Acquisition function value.
    """
    vmax = kwargs.get('vmax')
    Z = (mu - vmax) / sigma
    return np.where(sigma == 0, 0, sigma * (Z * scp.stats.norm.cdf(Z) + scp.stats.norm.pdf(Z)))
