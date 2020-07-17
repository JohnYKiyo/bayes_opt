from .bo import BayesOpt

from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
import matplotlib.pyplot as plt

def plot_history(BayesOpt_inst):
    """ plot baysian optimization history.

    Args:
        BayesOpt_inst (class): 

    Raises:
        TypeError: [description]
    """    
    if not isinstance(BayesOpt_inst,BayesOpt):
        raise TypeError('plot_history require BayesOpt')
    plt.plot(BayesOpt_inst.value_history,'.')
    if BayesOpt_inst.maximization:
        plt.plot(np.array([np.max(BayesOpt_inst.value_history[:i+1]) for i in range(BayesOpt_inst.n_trial)]),'-',label=BayesOpt_inst.acq.__name__)
    else:
        plt.plot(np.array([np.min(BayesOpt_inst.value_history[:i+1]) for i in range(BayesOpt_inst.n_trial)]),'-',label=BayesOpt_inst.acq.__name__)
    plt.xlabel('iter')
    plt.ylabel('value')
    plt.legend(loc='upper left',bbox_to_anchor=(1,1))