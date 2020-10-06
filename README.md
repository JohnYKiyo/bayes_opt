# Bayesian Optimization
## 1\. Overview
The baysian_optimization(gpbayesopt) package optimizes the black-box function. The algorithm is Baysian optimization based on the Gaussian process.  
The major differences from the famous Baysian Optimization package (GPyOpt) are as follows.    
1. You can design the acquisition function.   
2. You can design the kernel for use in the Gaussian Process.   
3. You can design an algorithm to find the next search point from the acquisition function.  

These will give you the flexibility to improve and experiment with Bayesian optimization.   
For example, if you want to deal with discretized search variables, you can change the kernel, acquisition function, and algorithm of next search point.
We hope that this will lead to the development of new Bayesian Optimization research.

## 2\. Installation

You can install the package from
[GitHub](https://github.com/JohnYKiyo/bayesian_optimization)

``` :sh
$ pip install git+https://github.com/JohnYKiyo/bayesian_optimization.git

```

Or install manualy

``` :sh
$ git clone https://github.com/JohnYKiyo/bayesian_optimization.git
$ cd bayesian_optimization
$ python setup.py install
```

## Dependencies

gpbayesopt requires:

- Python (>= 3.6)   
- Jax (>= 0.1.57)   
- Jaxlib (>= 0.1.37)   
- Scipy (>= 1.5.1)
- GaussianProcess (>= 0.5.0) https://github.com/JohnYKiyo/GaussianProcess.git   
This gpbayesopt package includes the works (Jax, Jaxlib) that are distributed in the Apache License 2.0.
