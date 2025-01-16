import numpy as np
from scipy.spatial.distance import cdist

def white_kernel(x1, x2, varSigma):
    if x2 is None:
        return varSigma*np.eye(x1.shape[0])
    else:
        return np.zeros(x1.shape[0], x2.shape[0])
    
def periodic_kernel(x1, x2, varSigma, period, lengthScale):
    if x2 is None:
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    return varSigma*np.exp(-(2*np.sin((np.pi/period)*d)**2)/lengthScale**2)

def get_coast_noise(DIM_MAP, scaling_factor=3, period=1, noise_level=0.01):
    x = np.linspace(0, DIM_MAP, DIM_MAP).reshape(-1,1)
    K = periodic_kernel(x, x, 1, period, 1) + white_kernel(x, None, noise_level)
    mu = np.zeros(x.shape)
    
    f = scaling_factor * np.random.multivariate_normal(mu.flatten(), K, 1)[0]
    return f