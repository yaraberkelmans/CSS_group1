from experiments.experiment import *


if __name__ == "__main__":
    # Predefined experiments are run via sim.py
    # Find below some examples of how to run them:

    # Example: run experiment to reproduce paper figue 1a
    run_experiment("Fig 1a", savepath="img/fig1a.png")

    # Example: run experiment for seed dependence
    # run_seed_dependence(savepath="img/seed_dependence.png")

    # Example: run experiment for convergences under varying alpha
    # run_experiment_h2a(savepath="img/experiment_h2a.png")

    # Example: run experiment for treating polarisation via noies
    # run_depolarisation_noise_increase(savepath="img/noise_depolarisation.png")
