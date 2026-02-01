from src.model import AgentBasedModel
from analysis import hdbscan_cluster_labels_xytheta, count_big_hdbscan_clusters
from visualization.plot import *
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


def run_find_polarised_r() -> None:
    """
    R sweep to find porlarised regimes
    """

    runs = 30

    R_op_values = np.linspace(0.05, 0.2, 50)
    polarisation_resultss = []

    for run in range(runs):
        print(f"--- Hysteresis Experiment Run {run+1}/{runs} ---")
        seed = np.random.randint(10000)
        polarisation_results = []

        for R_op in R_op_values:

            model = AgentBasedModel(
                N=100,
                T=2.5,
                dt=0.01,
                R_sp=0.15,
                R_op=R_op,
                sigma_sp=0.05,
                sigma_op=0.05,
                seed=seed,
                alpha=40.0,
                beta=10.0,
            )

            model.run()
            cross_cutt = model.cross_cutting_edge_fraction()
            assortativity = model.global_assortativity()
            opinion_var = model.opinion_variance()
            labels = hdbscan_cluster_labels_xytheta(model, theta_scale=1.0)
            n_big, size_by_id = count_big_hdbscan_clusters(labels, min_size=10)

            polarisation_results.append(
                np.array([R_op, cross_cutt, assortativity, opinion_var, n_big])
            )

        polarisation_results = np.array(polarisation_results)
        polarisation_resultss.append(polarisation_results)

    polarisation_resultss = np.array(polarisation_resultss)
    plot_polarisation_vs_Rop(
        polarisation_resultss, savepath="img/polarisation_vs_Rop.png"
    )


if __name__ == "__main__":
    # run_experiment("Fig 1a", savepath="img/fig1a.png")
    # run_experiment("Fig 1b")
    # run_experiment("Fig 2a")
    # run_experiment("Fig 2b")
    # run_experiment("Fig 3a")
    # run_hysteresis_experiment()
    run_find_polarised_r()
