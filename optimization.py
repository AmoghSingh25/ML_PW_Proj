from emukit.core import ParameterSpace, ContinuousParameter
from emukit.sensitivity.monte_carlo import MonteCarloSensitivity
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from config import PARAM_RANGES
from coastal_simulation import CoastalSimulation
import numpy as np
import GPy
from emukit.model_wrappers import GPyModelWrapper

class OptimizationManager:
    def __init__(self):
        self.simulation = CoastalSimulation()
        self.parameter_space = ParameterSpace([
            ContinuousParameter(key, *PARAM_RANGES[key]) 
            for key in PARAM_RANGES
        ])

    def sample_parameters(self, n_samples):
        return np.random.uniform(
            [PARAM_RANGES[key][0] for key in PARAM_RANGES],
            [PARAM_RANGES[key][1] for key in PARAM_RANGES],
            size=(n_samples, len(PARAM_RANGES))
        )

    def wrapped_run_sim(self, X):
        # Ensure X is a 2D array
        X = np.atleast_2d(X)
        results = []
        
        for x in X:
            wave_freq, wave_speed, wave_decay, wave_cutoff, wave_retreat_coeff, \
            wave_height, sand_pull, ground_pull, water_decay = x

            _, total_sand = self.simulation.run_sim(
                num_timesteps=100,
                wave_freq=wave_freq,
                wave_speed=wave_speed,
                wave_decay=wave_decay,
                wave_cutoff=wave_cutoff,
                wave_retreat_coeff=wave_retreat_coeff,
                wave_height=wave_height,
                sand_pull=sand_pull,
                ground_pull=ground_pull,
                water_decay=water_decay,
                plots=False
            )
            results.append([-total_sand])  # Wrap in a list to make it 2D

        return np.array(results)

    def run_sensitivity_analysis(self, X, Y):
        # Gaussian process model
        gp = GPy.models.GPRegression(X, Y)
        emukit_model = GPyModelWrapper(gp)

        # Sensitivity analysis
        sensitivity = MonteCarloSensitivity(emukit_model, self.parameter_space)
        sensitivity_results = sensitivity.compute_effects()
        main_effects = sensitivity_results[0]
        total_effects = sensitivity_results[1]

        return main_effects, total_effects

    def run_optimization(self, X, Y, n_iterations=20):
        # Initialize GP model
        gp = GPy.models.GPRegression(X, Y)
        emukit_model = GPyModelWrapper(gp)

        # Create and run Bayesian optimization loop
        bo_loop = BayesianOptimizationLoop(
            model=emukit_model,
            space=self.parameter_space
        )
        bo_loop.run_loop(self.wrapped_run_sim, n_iterations)

        # Return both the optimization results and the loop
        return bo_loop.loop_state