import numpy as np
from scipy.spatial.distance import cdist
from config import DIM_MAP

def white_kernel(x1, x2, varSigma):
    if x2 is None:
        return varSigma*np.eye(x1.shape[0])
    else:
        return np.zeros((x1.shape[0], x2.shape[0]))
    
def periodic_kernel(x1, x2, varSigma, period, lengthScale):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    return varSigma*np.exp(-(2*np.sin((np.pi/period)*d)**2)/lengthScale**2)

def rbf_kernel(x1, x2, varSigma, lengthScale):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    return varSigma*np.exp(-d**2/(2*lengthScale**2))

# def get_emulation_noise(scaling_factor=3, period=1, varSigma=0.01, lengthScale=1):
#     x = np.linspace(0, DIM_MAP, DIM_MAP).reshape(-1,1)
#     K = periodic_kernel(x, x, 1, 1, 5) + white_kernel(x, None, 0.05) + rbf_kernel(x, None, 1, 80)
#     mu = np.zeros(x.shape)
    
#     f = scaling_factor * np.random.multivariate_normal(mu.flatten(), K, 1)[0]
#     return f

def get_coast_noise_reproducible(scaling_factor=3, period=1, noise_level=0.01):
    rng = np.random.default_rng(19028)
    x = np.linspace(0, DIM_MAP, DIM_MAP).reshape(-1,1)
    K = periodic_kernel(x, x, 1, period, 1) + white_kernel(x, None, noise_level)
    mu = np.zeros(x.shape)
    
    f = scaling_factor * rng.multivariate_normal(mu.flatten(), K, 1)[0]
    return f
    
def get_coast_noise(scaling_factor=3, period=1, noise_level=0.01,gen_coast=False):
    if gen_coast:
        return get_coast_noise_reproducible(scaling_factor, period, noise_level)
    x = np.linspace(0, DIM_MAP, DIM_MAP).reshape(-1,1)
    K = periodic_kernel(x, x, 1, period, 1) + white_kernel(x, None, noise_level)
    mu = np.zeros(x.shape)
    
    f = scaling_factor * np.random.multivariate_normal(mu.flatten(), K, 1)[0]
    return f