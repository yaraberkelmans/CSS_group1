from random import seed
from model import AgentBasedModel
from analysis import hdbscan_cluster_labels_xytheta, count_big_hdbscan_clusters
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


def run_depolarisation_noise_increase(
    noise_range: np.ndarray = np.arange(0.01, 0.11, 0.01),
    iterations: int = 30,
    savepath: str = None,
) -> None:

    result = []

    for iteration in range(iterations):
        print(
            f"--- Depolarisation Noise Increase Iteration {iteration+1}/{iterations} ---"
        )
        seed = np.random.randint(10000)
        pre_treatment = []
        post_treatment = []

        for sigma_op in noise_range:

            # standard config
            model = AgentBasedModel(
                N=100,
                T=2.5,
                dt=0.01,
                R_sp=0.15,
                R_op=0.15,
                sigma_sp=0.05,
                sigma_op=0.05,
                seed=seed,
                alpha=40.0,
                beta=10.0,
            )
            model.run()
            pre_treatment_assortativity = model.global_assortativity()
            pre_treatment_opinion_variance = model.opinion_variance()
            pre_treatment.append(
                (pre_treatment_assortativity, pre_treatment_opinion_variance)
            )

            # treatment
            model.sigma_op = sigma_op
            model.sigma_sp = sigma_op
            model.run()
            post_treatment_assortativity = model.global_assortativity()
            post_treatment_opinion_variance = model.opinion_variance()
            post_treatment.append(
                (post_treatment_assortativity, post_treatment_opinion_variance)
            )

        result.append(np.array([pre_treatment, post_treatment]))

    result = np.array(result)
    print(result.shape)
    plot_depolarisation_vs_noise(result, noise_range=noise_range, savepath=savepath)


def run_depolarisation_ignore_connections(savepath: str = None) -> None:
    """
    Try depolarising by cutting random connections between agents, i.e. ignoring some edges
    """

    ignore_fraction_range = np.arange(0.05, 0.5, 0.05)
    iterations = 30

    result = []

    for iteration in range(iterations):
        print(
            f"--- Depolarisation Ignore Connections Iteration {iteration+1}/{iterations} ---"
        )

        pre_treatment = []
        post_treatment = []

        seed = np.random.randint(10000)

        for ignore_fraction in ignore_fraction_range:

            # standard config
            model = AgentBasedModel(
                N=100,
                T=2.5,
                dt=0.01,
                R_sp=0.15,
                R_op=0.15,
                sigma_sp=0.05,
                sigma_op=0.05,
                seed=seed,
                alpha=10.0,
                beta=10.0,
            )
            model.run()
            pre_treatment_assortativity = model.global_assortativity()
            pre_treatment_opinion_variance = model.opinion_variance()
            pre_treatment.append(
                (pre_treatment_assortativity, pre_treatment_opinion_variance)
            )

            # treatment - ignore some edges
            model.run(ignore_random_edges=ignore_fraction)
            post_treatment_assortativity = model.global_assortativity()
            post_treatment_opinion_variance = model.opinion_variance()
            post_treatment.append(
                (post_treatment_assortativity, post_treatment_opinion_variance)
            )

        result.append(np.array([pre_treatment, post_treatment]))

    result = np.array(result)
    print(result.shape)

    plot_depolarisation_vs_edge_removal(
        result, ignore_fraction_range=ignore_fraction_range, savepath=savepath
    )


def run_seed_dependence(savepath: str = None, n: int = 30) -> None:
    """
    Explore how different random seeds affect the final state of the system
    """
    reps = np.arange(n)
    result = []
    seeds = []

    for _ in reps:
        print(f"--- Seed Dependence Experiment Seed {_} ---")
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
            alpha=40.0,
            beta=10.0,
        )

        # run model
        model.run()

        # collect final metrics
        final_assortativity = model.global_assortativity()
        final_opinion_variance = model.opinion_variance()
        result.append((final_assortativity, final_opinion_variance))
        seeds.append(seed)

    result = np.array(result)
    seeds = np.array(seeds)
    plot_seed_dependence(result, seeds, savepath=savepath)

def run_fig3a_multiple_times(
    times=(0.55, 1.2, 1.9, 3.4),
    seed=None,
    base_savepath="img/fig3a",
):
    params = {"alpha": 40.0, "beta": 10.0}

    if seed is None:
        seed = np.random.randint(10000)
        print(f"Using seed: {seed}")

    for t in times:
        model = AgentBasedModel(
            N=100,
            T=t,
            dt=0.01,
            R_sp=0.15,
            R_op=0.15,
            sigma_sp=0.05,
            sigma_op=0.05,
            seed=seed,
            **params,
        )

        model.run()

        plot_snapshot_with_edges(
            model,
            title=f"Interaction network and opinions at T = {t}",
            savepath=f"{base_savepath}_T{str(t).replace('.', 'p')}.png",
        )



if __name__ == "__main__":
    # run_experiment("Fig 1a", savepath="img/fig1a.png")
    # run_experiment("Fig 1b")
    # run_experiment("Fig 2a")
    # run_experiment("Fig 2b")
    # run_experiment("Fig 3a")
    # run_find_polarised_r()
    # run_depolarisation_noise_increase(savepath="img/depolarisation_vs_noise.png")
    # run_depolarisation_ignore_connections(
    #    savepath="img/depolarisation_vs_edge_removal.png"
    # )
    # run_seed_dependence(savepath="img/seed_dependence.png", n=500)
    run_fig3a_multiple_times( times=(0.55, 1.2, 1.9, 3.4), seed=6836)

