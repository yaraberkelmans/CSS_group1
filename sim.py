from model import AgentBasedModel
from plot import *
import numpy as np


def run_experiment(case_name: str, savepath: str = None) -> None:

    print(f"--- Running Experiment: {case_name} ---")

    if case_name == "Fig 1a":
        params = {"alpha": 40.0, "beta": 10.0}
    elif case_name == "Fig 1b":
        params = {"alpha": 40.0, "beta": 40.0}
    elif case_name == "Fig 2a":
        params = {"alpha": 10.0, "beta": 50.0}
    elif case_name == "Fig 2b":
        params = {"alpha": 100.0, "beta": 50.0}
    elif case_name == "Fig 3a":
        params = {"alpha": 40.0, "beta": 10.0}
    else:
        print("Unknown case")
        return

    seed = np.random.randint(10000)

    model = AgentBasedModel(
        N=100,
        T=2.5,
        dt=0.01,
        R_sp=0.15,
        R_op=0.15,
        sigma_sp=0.05,
        sigma_op=0.05,
        seed=seed,
        **params,
    )

    model.run()

    if case_name == "Fig 3a":
        plot_snapshot_with_edges(
            model,
            title=f"{case_name} at T=2.5 (α={params['alpha']}, β={params['beta']})",
            savepath=savepath,
        )
    else:
        plot_snapshot(
            model,
            title=f"{case_name} (α={params['alpha']}, β={params['beta']})",
            savepath=savepath,
        )


def run_hysteresis_experiment():
    """Vary R_sp and R_op to observe hysteresis in assortativity measure."""

    # sweep R_sp and R_op from low to high and back
    R_values = np.linspace(0.05, 0.3, 10)
    assortativity_forward = []
    assortativity_backward = []
    seed = np.random.randint(10000)
    # Forward sweep
    for R in R_values:
        model = AgentBasedModel(
            N=100,
            T=2.5,
            dt=0.01,
            R_sp=R,
            R_op=R,
            sigma_sp=0.05,
            sigma_op=0.05,
            seed=seed,
            alpha=40.0,
            beta=10.0,
        )
        model.run()
        assortativity = model.global_assortativity()
        assortativity_forward.append(assortativity)
    # Backward sweep
    for R in reversed(R_values):
        model = AgentBasedModel(
            N=100,
            T=2.5,
            dt=0.01,
            R_sp=R,
            R_op=R,
            sigma_sp=0.05,
            sigma_op=0.05,
            seed=seed,
            alpha=40.0,
            beta=10.0,
        )
        model.run()
        assortativity = model.global_assortativity()
        assortativity_backward.append(assortativity)
    assortativity_backward.reverse()

    # Plot results
    plot_hysteresis(
        R_values,
        assortativity_forward,
        assortativity_backward,
    )


if __name__ == "__main__":
    # run_experiment("Fig 1a", savepath="img/fig1a.png")
    # run_experiment("Fig 1b")
    run_experiment("Fig 2a")
    # run_experiment("Fig 2b")
    # run_experiment("Fig 3a")
    # run_hysteresis_experiment()
