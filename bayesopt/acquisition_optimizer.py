from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp

from GaussianProcess.utils import transform_data


class BaseOptimizer(object):
    def __init__(self, bounds):
        self.bounds = np.atleast_2d(bounds)
        self.ndim = len(self.bounds)

    def __call__(self, gpr, acq, it):
        return self.optimize(gpr, acq, it)

    def optimize(self, gpr, acq, it):
        raise NotImplementedError("The optimize method is not implemented in the parent class.")


class Acquisition_L_BFGS_B_Optimizer(BaseOptimizer):
    def __init__(self, bounds, n_trial=2):
        """Optimizer for acquisition function by L-BFGS-B.

        Args:
            bounds (array-like):
                An array giving the search range for the parameter.
                :[[param1 min, param1 max],...,[param k min, param k max]]
            n_trial (int, optional): Number of trials to stabilize the L-BFGS-B. Defaults to 2.
        """
        super(Acquisition_L_BFGS_B_Optimizer, self).__init__(bounds)
        self.n_trial = n_trial

    def optimize(self, gpr, acq, it):
        vmax = np.max(gpr.Y_train)
        vmin = np.min(gpr.Y_train)
        loc = None
        value = None
        import scipy.optimize

        def Obj(x):
            mu, sigma = gpr.posterior_predictive(np.atleast_2d(x), return_std=True)
            return -1. * acq(mu, sigma, it=it, vmax=vmax, vmin=vmin).ravel()

        x_seeds = onp.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_trial, self.ndim))
        for xtry in x_seeds:
            res = scipy.optimize.fmin_l_bfgs_b(Obj,
                                               x0=xtry,
                                               bounds=self.bounds,
                                               approx_grad=True,
                                               maxiter=100)
            if (loc is None) or (res[1] < value):
                loc = res[0]
                value = res[1]
        return loc, value


class Acquisition_L_BFGS_B_LogOptimizer(BaseOptimizer):
    def __init__(self, bounds, n_trial=2):
        """Optimizer for acquisition function by L-BFGS-B.
        Args:
            bounds (array-like):
                An array giving the search range for the parameter.
                :[[param1 min, param1 max],...,[param k min, param k max]]
            n_trial (int, optional): Number of trials to stabilize the L-BFGS-B. Defaults to 2.
        """
        super(Acquisition_L_BFGS_B_LogOptimizer, self).__init__(bounds)
        self.n_trial = n_trial

    def optimize(self, gpr, acq, it):
        vmax = np.max(gpr.Y_train)
        vmin = np.min(gpr.Y_train)
        loc = None
        value = None
        import scipy.optimize

        def Obj(x):
            ex = np.power(10, x)
            mu, sigma = gpr.posterior_predictive(np.atleast_2d(ex), return_std=True)
            return -1. * acq(mu, sigma, it=it, vmax=vmax, vmin=vmin).ravel()

        x_seeds = onp.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_trial, self.ndim))
        for xtry in x_seeds:
            res = scipy.optimize.fmin_l_bfgs_b(Obj,
                                               x0=xtry,
                                               bounds=self.bounds,
                                               approx_grad=True,
                                               maxiter=100)
            if (loc is None) or (res[1] < value):
                loc = np.power(10, res[0])
                value = res[1]
        return loc, value


class Acquisition_SLSQP_Optimizer(BaseOptimizer):
    def __init__(self, bounds, n_trial=2):
        """Optimizer for acquisition function by SLSQP.
        Args:
            bounds (array-like):
                An array giving the search range for the parameter.
                :[[param1 min, param1 max],...,[param k min, param k max]]
            n_trial (int, optional): Number of trials to stabilize the SLSQP. Defaults to 2.
        """
        super(Acquisition_SLSQP_Optimizer, self).__init__(bounds)
        self.n_trial = n_trial

    def optimize(self, gpr, acq, it):
        vmax = np.max(gpr.Y_train)
        vmin = np.min(gpr.Y_train)
        loc = None
        value = None
        import scipy.optimize

        def Obj(x):
            mu, sigma = gpr.posterior_predictive(np.atleast_2d(x), return_std=True)
            return -1. * acq(mu, sigma, it=it, vmax=vmax, vmin=vmin).ravel()

        x_seeds = onp.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_trial, self.ndim))
        for xtry in x_seeds:
            res = scipy.optimize.fmin_slsqp(Obj,
                                            x0=xtry,
                                            bounds=self.bounds,
                                            iprint=0,
                                            full_output=True,
                                            iter=100)
            if (loc is None) or (res[1] < value):
                loc = res[0]
                value = res[1]
        return loc, value


class Acquisition_Grid_Optimizer(BaseOptimizer):
    def __init__(self, bounds, step):
        """Optimizer for acquisition function by Grid search.

        Args:
            bounds (array-like):
                An array giving the search range for the parameter.
                :[[param1 min, param1 max],...,[param k min, param k max]]
            step (array-like): Grid size. [param1 step size, param2 step size,..., param k step size]
        """
        super(Acquisition_Grid_Optimizer, self).__init__(bounds)
        self.step = step

    def optimize(self, gpr, acq, it):
        vmax = np.max(gpr.Y_train)
        vmin = np.min(gpr.Y_train)
        GS = GridSampler(self.bounds, self.step)
        mu_s, std_s = gpr.posterior_predictive(GS.grid, return_std=True)
        val = acq(mu_s, std_s, it=it, vmax=vmax, vmin=vmin).ravel()
        return GS.grid[np.argmax(val)], np.max(val)


class GridSampler(object):
    def __init__(self, bounds, step):
        self.__Xmin = np.atleast_2d(bounds)[:, 0]
        self.__Xmax = np.atleast_2d(bounds)[:, 1]
        # data dimention check
        if self.__Xmin.shape != self.__Xmax.shape:
            raise ValueError('Xmin,Xmax should be same size.')
        self.__ndim = len(self.__Xmin)

        # step size init
        self.__step = transform_data(step)
        if (self.__step.shape != (self.__ndim, 1)):
            if self.__step.shape[1] != 1:
                raise ValueError('step should be an 1-D array_like or a numerical value.')
            if self.__step.shape[0] == 1:
                self.__step = np.full_like(self.__Xmin, step)
            else:
                raise ValueError(f'step shape should be same shape of Xmin and Xmax: {self.__Xmin.shape}, but get{self.__step.shape}')

        # generate grid points
        d_list = tuple(np.arange(mi, ma, st) for mi, ma, st in zip(self.__Xmin, self.__Xmax, self.__step))
        self.grid = np.array(np.meshgrid(*d_list)).reshape(self.__ndim, -1).T

        # iterator
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

# def AcquisitionLBFGSBOptimizer(gpr, acq, it, bounds, n_trial=2):
#     bounds = np.atleast_2d(bounds)
#     vmax = np.max(gpr.Y_train)
#     vmin = np.min(gpr.Y_train)
#     ndim = len(bounds)
#     loc = None
#     value = None
#
#     import scipy.optimize
#     def Obj(x):
#         mu, sigma = gpr.posterior_predictive(np.atleast_2d(x), return_std=True)
#         return -1.*acq(mu, sigma, it=it, vmax=vmax, vmin=vmin).ravel()
#
#     x_seeds = onp.random.uniform(bounds[:,0],bounds[:,1], size=(n_trial,ndim))
#     for xtry in x_seeds:
#         res = scipy.optimize.fmin_l_bfgs_b(Obj,
#                                            x0=xtry,
#                                            bounds=bounds,
#                                            approx_grad=True,
#                                            maxiter=100)
#         if (loc is None) or (res[1] < value):
#             loc = res[0]
#             value = res[1]
#     return loc, value

# def AcquisitionSLSQPOptimizer(gpr, acq, it, bounds, n_trial=2):
#     bounds = np.atleast_2d(bounds)
#     vmax = np.max(gpr.Y_train)
#     vmin = np.min(gpr.Y_train)
#     ndim = len(bounds)
#     loc = None
#     value = None
#
#     import scipy.optimize
#     def Obj(x):
#         mu,sigma = gpr.posterior_predictive(np.atleast_2d(x),return_std=True)
#         return -1.*acq(mu,sigma, it=it, vmax=vmax, vmin=vmin).ravel()
#
#     x_seeds = onp.random.uniform(bounds[:,0],bounds[:,1], size=(n_trial,ndim))
#     for xtry in x_seeds:
#         res = scipy.optimize.fmin_slsqp(Obj,
#                                         x0=xtry,
#                                         bounds=bounds,
#                                         iprint=0,
#                                         full_output=True,
#                                         iter=100)
#         if (loc is None) or (res[1] < value):
#             loc = res[0]
#             value = res[1]
#     return loc, value

# def AcquisitionGridOptimizer(gpr, acq, it, bounds, step):
#     bounds = np.atleast_2d(bounds)
#     vmax = np.max(gpr.Y_train)
#     vmin = np.min(gpr.Y_train)
#
#     GS = GridSampler(bounds,step)
#     mu_s, std_s = gpr.posterior_predictive(GS.grid,return_std=True)
#     val = acq(mu_s, std_s, it=it, vmax=vmax, vmin=vmin).ravel()
#     return GS.grid[np.argmax(val)],np.max(val)

# class GridSampler(object):
#     def __init__(self, bounds, step):
#         self.__Xmin = np.atleast_2d(bounds)[:,0]
#         self.__Xmax = np.atleast_2d(bounds)[:,1]
#         ##data dimention check
#         if self.__Xmin.shape != self.__Xmax.shape :
#             raise ValueError('Xmin,Xmax should be same size.')
#         self.__ndim = len(self.__Xmin)
#
#         ##step size init
#         self.__step = transform_data(step)
#         if (self.__step.shape != (self.__ndim,1)):
#             if self.__step.shape[1] != 1:
#                 raise ValueError('step should be an 1-D array_like or a numerical value.')
#             if self.__step.shape[0] == 1:
#                 self.__step = np.full_like(self.__Xmin,step)
#             else:
#                 raise ValueError(f'step shape should be same shape of Xmin and Xmax: {self.__Xmin.shape}, but get{self.__step.shape}')
#
#         ##generate grid points
#         d_list = tuple(np.arange(mi,ma,st) for mi,ma,st in zip(self.__Xmin,self.__Xmax,self.__step))
#         self.grid = np.array(np.meshgrid(*d_list)).reshape(self.__ndim,-1).T
#
#         ###iterator###
#         self.__i = 0
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.__i == len(self.grid):
#             raise StopIteration()
#         ret = tuple(self.grid[self.__i])
#         self.__i += 1
#         return ret
#
#     def __call__(self):
#         return self.grid
