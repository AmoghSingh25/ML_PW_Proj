import GPy
from utils import white_kernel, periodic_kernel, rbf_kernel
import numpy as np
from scipy.spatial.distance import cdist


class CompositeKernel(GPy.kern.Kern):
    def __init__(self, input_dim, varSigma_white=0.05, varSigma_rbf=1.0, lengthScale_rbf=80,
                 varSigma_periodic=1.0, period=1, lengthScale_periodic=5, name="composite_kernel"):
        super().__init__(input_dim, active_dims=None, name=name)
        
        # Parameters for each kernel
        self.varSigma_white = GPy.core.parameterization.param.Param("varSigma_white", varSigma_white)
        self.varSigma_rbf = GPy.core.parameterization.param.Param("varSigma_rbf", varSigma_rbf)
        self.lengthScale_rbf = GPy.core.parameterization.param.Param("lengthScale_rbf", lengthScale_rbf)
        self.varSigma_periodic = GPy.core.parameterization.param.Param("varSigma_periodic", varSigma_periodic)
        self.period = GPy.core.parameterization.param.Param("period", period)
        self.lengthScale_periodic = GPy.core.parameterization.param.Param("lengthScale_periodic", lengthScale_periodic)
        
        # Link parameters for optimization
        self.link_parameters(self.varSigma_white, self.varSigma_rbf, self.lengthScale_rbf,
                             self.varSigma_periodic, self.period, self.lengthScale_periodic)

    def K(self, X, X2=None):
        """
        Full covariance matrix combining white, periodic, and RBF kernels.
        """
        return (white_kernel(X, X2, self.varSigma_white) +
                periodic_kernel(X, X2, self.varSigma_periodic, self.period, self.lengthScale_periodic) +
                rbf_kernel(X, X2, self.varSigma_rbf, self.lengthScale_rbf)) 

    def Kdiag(self, X):
        """
        Diagonal of the covariance matrix.
        """
        return np.diag(self.K(X))

    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        Compute gradients of the kernel parameters with respect to the log-likelihood.
        """
        # White kernel gradient
        self.varSigma_white.gradient = np.sum(dL_dK * white_kernel(X, X2, 1.0))

        # Periodic kernel gradients
        dK_dvarSigma_periodic = periodic_kernel(X, X2, 1.0, self.period, self.lengthScale_periodic)
        # Ensure X and X2 are 2D arrays
        X = np.atleast_2d(X)
        X2 = np.atleast_2d(X2) if X2 is not None else X  # Use X for self-similarity if X2 is None

        # Proceed with computation
        dK_dlengthScale_periodic = (2 * dK_dvarSigma_periodic *
                                    (2 * np.sin(np.pi / self.period * cdist(X, X2)) ** 2) /
                                    (self.lengthScale_periodic ** 3))

        dK_dperiod = (2 * np.pi / self.period ** 2) * dK_dvarSigma_periodic * np.sin(
            2 * np.pi / self.period * cdist(X, X2)) * cdist(X, X2)

        self.varSigma_periodic.gradient = np.sum(dL_dK * dK_dvarSigma_periodic)
        self.lengthScale_periodic.gradient = np.sum(dL_dK * dK_dlengthScale_periodic)
        self.period.gradient = np.sum(dL_dK * dK_dperiod)

        # RBF kernel gradients
        dK_dvarSigma_rbf = rbf_kernel(X, X2, 1.0, self.lengthScale_rbf)
        dK_dlengthScale_rbf = (dK_dvarSigma_rbf * cdist(X, X2) ** 2) / (self.lengthScale_rbf ** 3)

        self.varSigma_rbf.gradient = np.sum(dL_dK * dK_dvarSigma_rbf)
        self.lengthScale_rbf.gradient = np.sum(dL_dK * dK_dlengthScale_rbf)

    def update_gradients_diag(self, dL_dKdiag, X):
        """
        Compute gradients with respect to the diagonal of the covariance matrix.
        """
        self.update_gradients_full(np.diag(dL_dKdiag), X)

class Emulator:
    def __init__(self):
        """
        Initialize the emulator with default kernel parameters.
        """
        self.gp = None

    def train_emulator(self, X, Y):
        """
        Train a Gaussian Process emulator.
        """
        # Instantiate the composite kernel with default parameters
        kernel = CompositeKernel(input_dim=X.shape[1])
        
        # Initialize the Gaussian Process model first
        self.gp = GPy.models.GPRegression(X, Y, kernel) 
        
        # Now compute the kernel matrix (after gp is initialized)
        K = self.gp.kern.K(X)  # Compute the kernel matrix
        jitter = 1e-4 * np.eye(K.shape[0])  # Create a small diagonal matrix
        K += jitter  # Add jitter to the kernel matrix

        self.gp.optimize()  # Optimize GP hyperparameters
        return self.gp

    def predict(self, X_new):
        """
        Predict using the trained emulator.
        """
        if self.gp is None:
            raise ValueError("The emulator must be trained before making predictions.")
        return self.gp.predict(X_new)
