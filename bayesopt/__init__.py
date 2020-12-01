import GaussianProcess
from . import acquisition
from . import acquisition_optimizer
from .bo import BayesOpt
from .plotter import plot_history
from GaussianProcess import kernel, utils, GPR, metric

__all__ = ['GaussianProcess', 'acquisition', 'acquisition_optimizer', 'BayesOpt', 'kernel', 'utils', 'GPR', 'metric', 'plot_history']
