from .bo import BayesOpt

from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
import matplotlib.pyplot as plt

def plot_history(BayesOpt_obj):
    if not isinstance(BayesOpt_obj,BayesOpt):
        raise TypeError('plot_history require BayesOpt')
    plt.plot(BayesOpt_obj.value_history,'.')
    if BayesOpt_obj.maximization:
        plt.plot(np.array([np.max(BayesOpt_obj.value_history[:i+1]) for i in range(BayesOpt_obj.n_trial)]),'-',label=BayesOpt_obj.acq.__name__)
    else:
        plt.plot(np.array([np.min(BayesOpt_obj.value_history[:i+1]) for i in range(BayesOpt_obj.n_trial)]),'-',label=BayesOpt_obj.acq.__name__)
    plt.xlabel('iter')
    plt.ylabel('value')
    plt.legend()