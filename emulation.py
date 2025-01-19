import GPy
from utils import get_emulation_noise

# Emulation class
class Emulator:
    def __init__(self):
        """
        Initialize the emulator 
        """
        kernel_params = {}
        self.kernel_params = kernel_params
        self.gp = None

    def train_emulator(self, X, Y):
        """
        Train a Gaussian Process emulator.
        """
        kernel = GPy.kern.CustomKernel(input_dim=X.shape[1], kernel_function=self.kernel_function)
        self.gp = GPy.models.GPRegression(X, Y, kernel)
        self.gp.optimize()  # Optimize GP hyperparameters
        return self.gp

    def kernel_function(self, x1, x2=None):
        """
        Custom kernel function wrapper.
        """
        return get_emulation_noise(x1, x2, **self.kernel_params)

    def predict(self, X_new):
        """
        Predict using the trained emulator.
        """
        if self.gp is None:
            raise ValueError("The emulator must be trained before making predictions.")
        return self.gp.predict(X_new)