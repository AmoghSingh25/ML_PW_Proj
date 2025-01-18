import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from emukit.core import ParameterSpace, ContinuousParameter
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from config import (
    PARAM_RANGES, OBSTACLE_COORDS, WAVE_FREQ, WAVE_SPEED, WAVE_DECAY, WAVE_CUTOFF,
    WAVE_RETREAT_COEFF, WAVE_HEIGHT, SAND_PULL, GROUND_PULL, WATER_DECAY,
    WAVE_VOL, WAVE_AMPLITUDE, WAVE_SPREAD, DIM_MAP
)
from simulation3D import Simulation3D
import numpy as np
import GPy
from emukit.model_wrappers import GPyModelWrapper

class OptimizationManager3D:
    def __init__(self):
        self.num_obstacles = len(OBSTACLE_COORDS)

        # Define parameter space for barrier coordinates
        self.parameter_space = ParameterSpace([
            ContinuousParameter(f"obstacle_x{i+1}", *PARAM_RANGES["obstacle_coords"][0])
            for i in range(self.num_obstacles)
        ] + [
            ContinuousParameter(f"obstacle_y{i+1}", *PARAM_RANGES["obstacle_coords"][1])
            for i in range(self.num_obstacles)
        ])

    def sample_parameters(self, n_samples):
        # Generate random samples for barrier coordinates
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

        return np.hstack((x_samples, y_samples))

    def wrapped_run_sim(self, X):
        # Ensure X is a 2D array
        X = np.atleast_2d(X)
        results = []

        for x in X:
            # Extract barrier coordinates and cast to integers
            obstacle_coords = [(int(x[i]), int(x[self.num_obstacles + i])) for i in range(self.num_obstacles)]

            # Initialize 3D simulation
            sim = Simulation3D(
                wave_freq=WAVE_FREQ,
                wave_speed=WAVE_SPEED,
                wave_decay=WAVE_DECAY,
                wave_cutoff=WAVE_CUTOFF,
                wave_retreat_coeff=WAVE_RETREAT_COEFF,
                wave_height=WAVE_HEIGHT,
                sand_pull=SAND_PULL,
                ground_pull=GROUND_PULL,
                water_decay=WATER_DECAY,
                wave_vol=WAVE_VOL,
                wave_amplitude=WAVE_AMPLITUDE,
                wave_spread=WAVE_SPREAD,
                obstacle_coords=obstacle_coords,
                dim_map=DIM_MAP
            )

            # Run simulation
            sim.run_sim(num_timesteps=100)

            # Calculate total erosion (sum of sand loss across all layers)
            sand_before = np.sum(sim.coast_map[:, :, 1])  # Before erosion
            sand_after = np.sum(sim.coast_map_3D[:, :, :, 1])  # After erosion in 3D
            erosion_diff = sand_before - sand_after  # Minimize this difference

            results.append([-erosion_diff])  # Negative for minimization

        return np.array(results)

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
