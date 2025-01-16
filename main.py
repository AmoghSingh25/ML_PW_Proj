from optimization import OptimizationManager
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Initialize optimization manager
    opt_manager = OptimizationManager()
    
    # Initial sampling
    n_samples = 10
    param_samples = opt_manager.sample_parameters(n_samples)
    
    # Run initial simulations
    outcomes = []
    for params in param_samples:
        _, outcome = opt_manager.simulation.run_sim(
            num_timesteps=100,
            wave_freq=params[0],
            wave_speed=params[1],
            wave_decay=params[2],
            wave_cutoff=params[3],
            wave_retreat_coeff=params[4],
            wave_height=params[5],
            sand_pull=params[6],
            ground_pull=params[7],
            water_decay=params[8]
        )
        outcomes.append(outcome)

    # Convert to numpy arrays for EmuKit
    X = np.array(param_samples)
    Y = np.array(outcomes).reshape(-1, 1)

    # Run sensitivity analysis
    main_effects, total_effects = opt_manager.run_sensitivity_analysis(X, Y)
    print("Main effects:", main_effects)
    print("Total effects:", total_effects)

    # Run Bayesian optimization
    results = opt_manager.run_optimization(X, Y)
    
    # Get best parameters and outcome
    best_parameters = results.X[-1]
    best_outcome = results.Y[-1]
    
    print("Best parameters:", best_parameters)
    print("Best outcome:", best_outcome)

    # Visualize final result
    final_state, _ = opt_manager.simulation.run_sim(
        num_timesteps=100,
        wave_freq=best_parameters[0],
        wave_speed=best_parameters[1],
        wave_decay=best_parameters[2],
        wave_cutoff=best_parameters[3],
        wave_retreat_coeff=best_parameters[4],
        wave_height=best_parameters[5],
        sand_pull=best_parameters[6],
        ground_pull=best_parameters[7],
        water_decay=best_parameters[8],
        plots=True
    )
    plt.imshow(final_state)
    plt.show()

if __name__ == "__main__":
    main()