from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
import jax.scipy as scp
from jax import jit
import numpy as onp

from .gp.gp.utils import transform_data

def AcquisitionLBFGSBOptimizer(gpr, acq, it, bounds):
    bounds = np.atleast_2d(bounds)
    vmax = np.max(gpr.Y_train)
    vmin = np.min(gpr.Y_train)

    import scipy.optimize 
    def Obj(x):
        mu, sigma = gpr.posterior_predictive(x, return_std=True)
        return -1.*acq(mu, sigma, it=it, vmax=vmax, vmin=vmin).ravel()

    res = scipy.optimize.fmin_l_bfgs_b(Obj,
                                       x0=onp.random.uniform(bounds[:,0],bounds[:,1]), 
                                       bounds=bounds, 
                                       approx_grad=True, 
                                       maxiter=100)
    return res[0],res[1]

def AcquisitionSLSQPOptimizer(gpr, acq, it, bounds):
    bounds = np.atleast_2d(bounds)
    vmax = np.max(gpr.Y_train)
    vmin = np.min(gpr.Y_train)

    import scipy.optimize  
    def Obj(x):
        mu,sigma = gpr.posterior_predictive(np.atleast_2d(x),return_std=True)
        return -1.*acq(mu,sigma, it=it, vmax=vmax, vmin=vmin).ravel()
    res = scipy.optimize.fmin_slsqp(Obj, 
                                    x0=onp.random.uniform(bounds[:,0],bounds[:,1]), 
                                    bounds=bounds, 
                                    iprint=0, 
                                    full_output=True, 
                                    iter=100)

    return res[0],res[1]

def AcquisitionGridOptimizer(gpr, acq, it, bounds, step):
    bounds = np.atleast_2d(bounds)
    vmax = np.max(gpr.Y_train)
    vmin = np.min(gpr.Y_train)

    GS = GridSampler(bounds,step)
    mu_s, std_s = gpr.posterior_predictive(GS.grid,return_std=True)
    val = -1.*acq(mu_s, std_s, it=it, vmax=vmax, vmin=vmin).ravel()
    return GS.grid[np.argmin(val)],np.min(val)

class GridSampler(object):
    def __init__(self, bounds, step):
        self.__Xmin = np.atleast_2d(bounds)[:,0]
        self.__Xmax = np.atleast_2d(bounds)[:,1]
        ##data dimention check
        if self.__Xmin.shape != self.__Xmax.shape :
            raise ValueError('Xmin,Xmax should be same size.')
        self.__ndim = len(self.__Xmin)

        ##step size init
        self.__step = bayesopt.utils.transform_data(step)
        if (self.__step.shape != (self.__ndim,1)):
            if self.__step.shape[1] != 1:
                raise ValueError('step should be an 1-D array_like or a numerical value.')
            if self.__step.shape[0] == 1:
                self.__step = np.full_like(self.__Xmin,step)
            else:
                raise ValueError(f'step shape should be same shape of Xmin and Xmax: {self.__Xmin.shape}, but get{self.__step.shape}')

        ##generate grid points
        d_list = tuple(np.arange(mi,ma,st) for mi,ma,st in zip(self.__Xmin,self.__Xmax,self.__step))
        self.grid = np.array(np.meshgrid(*d_list)).reshape(self.__ndim,-1).T

        ###iterator###
        self.__i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__i == len(self.grid):
            raise StopIteration()
        ret = tuple(self.grid[self.__i])
        self.__i += 1
        return ret

    def __call__(self):
        return self.grid