from model import AgentBasedModel
from plot import plot_snapshot, plot_snapshot_with_edges 
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

    model = AgentBasedModel(
        N=100,
        T=2.5,
        dt=0.01,
        R_sp=0.15,
        R_op=0.15,
        sigma_sp=0.05,
        sigma_op=0.05,
        **params,
    )

    seed = np.random.randint(10000)
    model.run(seed=seed)

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



if __name__ == "__main__":
    run_experiment("Fig 1a", savepath="img/fig1a.png")
    # run_experiment("Fig 1b")
    # run_experiment("Fig 2a")
    # run_experiment("Fig 2b")
    # run_experiment("Fig 3a")
