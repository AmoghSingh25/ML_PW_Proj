import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

import numpy as np
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.optimization import RandomSearchAcquisitionOptimizer
from emukit.model_wrappers import GPyModelWrapper
from simulation import simulation
from config import PARAM_RANGES, OBSTACLE_COORDS, DIM_MAP, WAVE_DECAY,WAVE_CUTOFF, WAVE_RETREAT_COEFF, WAVE_VOL, WAVE_SPREAD, SAND_PULL, GROUND_PULL, WATER_DECAY, WAVE_FREQ, WAVE_SPEED, WAVE_HEIGHT, WAVE_AMPLITUDE

class OptimizationManager:
    def __init__(self, emulator):
        self.emulator = emulator  # Pass in an emulator instance
        self.num_obstacles = len(OBSTACLE_COORDS)

        # Create parameter space, focusing only on obstacle placement
        self.parameter_space = ParameterSpace([
            ContinuousParameter(f"obstacle_x{i+1}", *PARAM_RANGES["obstacle_coords"][0])
            for i in range(self.num_obstacles)
        ] + [
            ContinuousParameter(f"obstacle_y{i+1}", *PARAM_RANGES["obstacle_coords"][1])
            for i in range(self.num_obstacles)
        ])

    def wrapped_run_sim(self, X):
        """
        Evaluate the simulation for given parameters, only optimizing obstacle placement.
        """
        X = np.atleast_2d(X)
        results = []

        for x in X:
            # Extract obstacle coordinates
            obstacle_coords = [(int(x[i]), int(x[self.num_obstacles + i])) for i in range(self.num_obstacles)]

            # Initialize simulation with fixed wave parameters and optimized obstacles
            sim = simulation(
                obstacle_coords=obstacle_coords,
                dim_map=DIM_MAP,
                wave_freq=WAVE_FREQ,
                wave_speed=WAVE_SPEED,
                wave_height=WAVE_HEIGHT,
                wave_amplitude=WAVE_AMPLITUDE,
                wave_vol=WAVE_VOL,
                wave_spread=WAVE_SPREAD,
                sand_pull=SAND_PULL,
                ground_pull=GROUND_PULL,
                water_decay=WATER_DECAY,
                wave_decay=WAVE_DECAY,
                wave_cutoff=WAVE_CUTOFF,
                wave_retreat_coeff=WAVE_RETREAT_COEFF
            )

            # Run simulation and compute objective
            _, _, before_erosion, after_erosion = sim.run_sim(num_timesteps=100)
            erosion_diff = before_erosion - after_erosion
            results.append([erosion_diff])

        return np.array(results)

    def run_optimization(self, X, Y, n_iterations=20):
        """
        Perform Bayesian Optimization using the emulator.
        """
        # Normalise input data
        X_mean, X_std = X.mean(axis=0), X.std(axis=0)
        X_std[X_std == 0] = 1 # no division by zero
        X_normalized = (X - X_mean) / X_std

        Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)
        Y_std[Y_std == 0] = 1 # no division by zero
        Y_normalized = (Y - Y_mean) / Y_std

        # Train emulator
        self.emulator.train_emulator(X_normalized, Y_normalized)

        # Wrap emulator for Bayesian Optimization
        emukit_model = GPyModelWrapper(self.emulator.gp)

        # Run Bayesian Optimization Loop
        bo_loop = BayesianOptimizationLoop(
            model=emukit_model,
            space=self.parameter_space,
            acquisition_optimizer=RandomSearchAcquisitionOptimizer(self.parameter_space),
        )
        bo_loop.run_loop(self.wrapped_run_sim, n_iterations)

        return bo_loop.loop_state

    def sample_parameters(self, n_samples):
        """
        Generate random samples from the parameter space, only for obstacle placement.
        """
        # Sample obstacle coordinates
        x_ranges = [PARAM_RANGES["obstacle_coords"][0] for _ in range(self.num_obstacles)]
        y_ranges = [PARAM_RANGES["obstacle_coords"][1] for _ in range(self.num_obstacles)]

        x_samples = np.random.uniform(
            [r[0] for r in x_ranges],
            [r[1] for r in x_ranges],
            size=(n_samples, len(x_ranges))
        )

        y_samples = np.random.uniform(
            [r[0] for r in y_ranges],
            [r[1] for r in y_ranges],
            size=(n_samples, len(y_ranges))
        )

        # Concatenate all samples
        return np.hstack((x_samples, y_samples))
