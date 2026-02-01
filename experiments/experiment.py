import numpy as np
import matplotlib.pyplot as plt
import tqdm
from random import seed
from src.analysis import hdbscan_cluster_labels_xytheta, count_big_hdbscan_clusters
from src.model import AgentBasedModel
from visualization.plot import *

# =========================================================================================
# Helper functions
# =========================================================================================


def run_trajectory(params, sample_rate=10):
    """Runs a single simulation and samples assortativity at given rate."""
    model = AgentBasedModel(**params)

    times = []
    assortativities = []
    steps = int(model.T / model.dt)

    rng = np.random.default_rng()

    # t=0
    times.append(0)
    assortativities.append(model.global_assortativity())

    for step in range(1, steps + 1):
        model.step(rng)

        if step % sample_rate == 0:
            times.append(step * model.dt)
            assortativities.append(model.global_assortativity())

    return np.array(times), np.array(assortativities)


def get_convergence_time(params, target_r=0.9, max_steps=3000):
    """Runs a simulation until the global assortativity reaches the target value or max steps."""
    model = AgentBasedModel(**params)
    rng = np.random.default_rng()

    for step in range(max_steps):
        model.step(rng)

        if step % 10 == 0:
            r = model.global_assortativity()
            if r >= target_r:
                return step * model.dt


# =========================================================================================
# 1. test H1: phase transition
# =========================================================================================


# h1a: assortativity vs R_op (scatter plot)
def run_experiment_h1a_scatter(savepath=None):
    """Run experiment for H1a: scatter plot of assortativity vs R_op."""
    R_op_values = np.linspace(0.05, 0.25, 50)
    n_simulations = 20

    base_params = {
        "N": 100,
        "T": 10.0,
        "dt": 0.01,
        "alpha": 40.0,
        "beta": 10.0,
        "R_sp": 0.15,
        "sigma_sp": 0.05,
        "sigma_op": 0.05,
    }

    x_coords = []  # R_op
    y_coords = []  # Assortativity

    for r_op in tqdm.tqdm(R_op_values):
        current_params = base_params.copy()
        current_params["R_op"] = r_op

        for i in range(n_simulations):
            seed = np.random.randint(0, 100000)
            current_params["seed"] = seed

            model = AgentBasedModel(**current_params)
            model.run()

            r = model.global_assortativity()
            x_coords.append(r_op)
            y_coords.append(r)

    # plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, s=20, alpha=0.2, c="black", marker="o")

    plt.title(
        f"Discontinuous Phase Transition in Global Assortativity vs $R_{{op}}$\n(N={base_params['N']}, α={base_params['alpha']}, β={base_params['beta']}, σ={base_params['sigma_op']})"
    )
    plt.xlabel("Opinion Interaction Radius ($R_{op}$)", fontsize=12)
    plt.ylabel("Global Assortativity ($r$)", fontsize=12)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.show()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)


# h1a: assortativity vs R_op (histogram)
def run_experiment_h1a_histogram(savepath=None):
    """Run experiment for H1a: histogram of assortativity vs R_op."""
    R_op_list = [0.15, 0.16, 0.17, 0.18]
    n_simulations = 50

    base_params = {
        "N": 100,
        "dt": 0.01,
        "alpha": 40.0,
        "beta": 10.0,
        "R_sp": 0.15,
        "sigma_sp": 0.05,
        "sigma_op": 0.05,
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, r_op in enumerate(R_op_list):
        results = []
        # run n simulations
        for _ in tqdm.tqdm(range(n_simulations), leave=False):
            seed = np.random.randint(0, 100000)

            current_params = base_params.copy()
            current_params["R_op"] = r_op
            current_params["seed"] = seed

            model = AgentBasedModel(T=10.0, **current_params)
            model.run()
            results.append(model.global_assortativity())

        ax = axes[idx]
        ax.hist(
            results,
            bins=15,
            range=(-0.1, 1.1),
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
        )

        ax.set_title(f"$R_{{op}} = {r_op}$", fontsize=14, fontweight="bold")

        # calculate bimodality coefficient
        from scipy.stats import skew, kurtosis

        s = skew(results)
        k = kurtosis(results, fisher=True) + 3
        bc = (s**2 + 1) / k
        ax.text(
            0.05,
            0.9,
            f"BC={bc:.2f}\n(>0.55 implies bimodal)",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    fig.suptitle(
        f"Evolution of Assortativity Distributions Across the Critical Transition Regime\n(N={base_params['N']}, α={base_params['alpha']}, β={base_params['beta']}, σ={base_params['sigma_op']})",
        fontsize=16,
    )
    fig.text(0.5, 0.04, "Global Assortativity (r)", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "Count", va="center", rotation="vertical", fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)


# h1b: opinions vs R_op
def run_experiment_h1b(savepath=None):
    """Run experiment for H1b: opinions vs R_op."""
    R_op_values = np.linspace(0.05, 0.25, 50)
    n_simulations = 10

    base_params = {
        "N": 100,
        "T": 10.0,
        "dt": 0.01,
        "alpha": 40.0,
        "beta": 10.0,
        "R_sp": 0.15,
        "sigma_sp": 0.05,
        "sigma_op": 0.05,
    }

    # save plot points
    x_coords = []  # R_op
    y_coords = []  # theta

    for r_op in tqdm.tqdm(R_op_values):
        current_params = base_params.copy()
        current_params["R_op"] = r_op

        for i in range(n_simulations):
            seed = np.random.randint(0, 100000)
            current_params["seed"] = seed

            model = AgentBasedModel(**current_params)
            model.run()

            # record all agents opinions at steady state
            current_opinions = [agent.theta[0] for agent in model.agents]

            x_coords.extend([r_op] * len(current_opinions))
            y_coords.extend(current_opinions)

    # plotting
    plt.figure(figsize=(12, 7))

    plt.scatter(x_coords, y_coords, s=20, alpha=0.1, c="black", marker="o")
    plt.axhline(y=0, color="gray", linestyle=":", alpha=0.3)

    plt.title(
        f"Steady States distribution vs $R_{{op}}$\n(N={base_params['N']}, α={base_params['alpha']}, β={base_params['beta']}, σ={base_params['sigma_op']})",
        fontsize=14,
    )
    plt.xlabel("Opinion Interaction Radius ($R_{op}$)", fontsize=12)
    plt.ylabel("Individual Agent Opinions ($\\theta$)", fontsize=12)

    plt.ylim(-1.1, 1.1)
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)


# =========================================================================================
# 2. test H2: speed of emergence varying alpha and beta
# =========================================================================================
def run_experiment_h2a(savepath=None):
    """Compares speed of convergence for varying alpha values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    param_str = (
        r"$N=100, \beta=10.0, R_{sp}=0.15, R_{op}=0.15, \sigma_{sp}=\sigma_{op}=0.05$"
    )

    # plot A: Assortativity vs Time
    alpha_values_traj = [5.0, 40.0, 80.0]
    colors_traj = ["#1f77b4", "#ff7f0e", "#d62728"]

    base_params_traj = {
        "N": 100,
        "T": 10.0,
        "dt": 0.01,
        "beta": 10.0,
        "R_sp": 0.15,
        "R_op": 0.15,
        "sigma_sp": 0.05,
        "sigma_op": 0.05,
    }

    for idx, alpha in enumerate(alpha_values_traj):
        current_params = base_params_traj.copy()
        current_params["alpha"] = alpha

        all_runs = []
        for _ in range(20):  # run 20 times for average
            times, r_curve = run_trajectory(current_params, sample_rate=20)
            all_runs.append(r_curve)

        min_len = min(len(r) for r in all_runs)
        all_runs = np.array([r[:min_len] for r in all_runs])
        times = times[:min_len]

        mean_curve = np.mean(all_runs, axis=0)
        std_curve = np.std(all_runs, axis=0)

        ax1.plot(
            times,
            mean_curve,
            label=r"$\alpha$=" + f"{alpha}",
            color=colors_traj[idx],
            lw=2.5,
        )
        ax1.fill_between(
            times,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=colors_traj[idx],
            alpha=0.15,
        )

    ax1.set_title(
        f"(a) Global Assortativity vs. Time (Varying $\\alpha$)\n{param_str}",
        fontsize=11,
        pad=10,
    )
    ax1.set_xlabel("Time ($t$)", fontsize=12)
    ax1.set_ylabel("Global Assortativity ($r$)", fontsize=12)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=11, loc="lower right", title="Opinion Strength")

    # plot B: Convergence Time vs Opinion Strength alpha
    alpha_values_speed = np.linspace(1.0, 100, 10)
    convergence_times = []
    errors = []

    base_params_speed = {
        "N": 100,
        "dt": 0.01,
        "beta": 10.0,
        "R_sp": 0.15,
        "R_op": 0.15,
        "sigma_sp": 0.05,
        "sigma_op": 0.05,
    }

    target_r = 0.8

    for alpha in tqdm.tqdm(alpha_values_speed, leave=False):
        current_params = base_params_speed.copy()
        current_params["alpha"] = alpha

        times = []
        for _ in range(20):
            t = get_convergence_time(current_params, target_r=target_r)
            if t is not None:
                times.append(t)
            else:
                times.append(10.0)

        convergence_times.append(np.mean(times))
        errors.append(np.std(times))

    ax2.errorbar(
        alpha_values_speed,
        convergence_times,
        yerr=errors,
        fmt="-o",
        color="darkgreen",
        ecolor="darkgreen",
        capsize=4,
        lw=2,
        markersize=6,
    )

    ax2.set_title(
        f"(b) Convergence Time vs. Opinion Strength\n{param_str}, Target $r={target_r}$",
        fontsize=11,
        pad=10,
    )
    ax2.set_xlabel(r"Opinion Strength ($\alpha$)", fontsize=12)
    ax2.set_ylabel("Convergence Time ($T_{conv}$)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)


def run_experiment_h2b(savepath=None):
    """Compares speed of convergence for varying beta values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    param_str = r"$N=100, \alpha=40.0, R_{sp}=0.15, R_{op}=0.15, \sigma=0.05$"

    # plot A: Assortativity vs Time
    beta_values_traj = [5, 10, 50]
    colors_traj = ["#1f77b4", "#ff7f0e", "#d62728"]

    base_params = {
        "N": 100,
        "T": 10.0,
        "dt": 0.01,
        "alpha": 40.0,
        "R_sp": 0.15,
        "R_op": 0.15,
        "sigma_sp": 0.05,
        "sigma_op": 0.05,
    }

    for idx, beta in enumerate(beta_values_traj):
        current_params = base_params.copy()
        current_params["beta"] = beta

        all_runs = []
        for _ in range(20):
            times, r_curve = run_trajectory(current_params, sample_rate=20)
            all_runs.append(r_curve)

        min_len = min(len(r) for r in all_runs)
        all_runs = np.array([r[:min_len] for r in all_runs])
        times = times[:min_len]

        mean_curve = np.mean(all_runs, axis=0)
        std_curve = np.std(all_runs, axis=0)

        ax1.plot(
            times,
            mean_curve,
            label=r"$\beta$=" + f"{beta}",
            color=colors_traj[idx],
            lw=2.5,
        )
        ax1.fill_between(
            times,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=colors_traj[idx],
            alpha=0.15,
        )

    ax1.set_title(
        f"(a) Global Assortativity vs. Time (Varying $\\beta$)\n{param_str}",
        fontsize=11,
        pad=10,
    )
    ax1.set_xlabel("Time ($t$)", fontsize=12)
    ax1.set_ylabel("Global Assortativity ($r$)", fontsize=12)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=11, loc="lower right", title="Social Strength")

    # part B: Convergence Time vs Beta
    beta_values_speed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
    convergence_times = []
    errors = []

    target_r = 0.8

    for beta in tqdm.tqdm(beta_values_speed, leave=False):
        current_params = base_params.copy()
        current_params["beta"] = beta

        times = []
        for _ in range(50):
            t = get_convergence_time(current_params, target_r=target_r)
            if t is not None:
                times.append(t)
            else:
                times.append(10.0)  # manually set max time if not converged

        convergence_times.append(np.mean(times))
        errors.append(np.std(times))

    ax2.errorbar(
        beta_values_speed,
        convergence_times,
        yerr=errors,
        fmt="-o",
        color="purple",
        ecolor="purple",
        capsize=4,
        lw=2,
        markersize=6,
    )

    ax2.set_title(
        f"(b) Convergence Time vs. Social Strength\n{param_str}, Target $r={target_r}$",
        fontsize=11,
        pad=10,
    )
    ax2.set_xlabel(r"Social Strength ($\beta$)", fontsize=12)
    ax2.set_ylabel("Convergence Time ($T_{conv}$)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)


# =========================================================================================
# 3. Predefined experiments for paper figures
# =========================================================================================


def run_experiment(case_name: str, savepath: str = None) -> None:
    """Runs predefined experiments corresponding to figures in the paper."""

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


# ========================================================================================
# 4. Additional experiments for polarisation treatment
# ========================================================================================


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
    """Uses noise increase as treatment to see whether depolarisation occurs."""

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


# ========================================================================================
# 5. Various additional experiments
# ========================================================================================


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
    """Plot Fig 3a snapshots at multiple times for a given seed."""

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


def run_r5_noise_sweep_dict(
    sigma_op_values: np.ndarray = np.linspace(0.0, 0.2, 21),
    runs=20,
    seed0=0,
    theta_scale=1.0,
    min_cluster_size=10,
    min_size_big=10,
    savepath=None,
):
    """Sweep across noise values for R5 experiment. Does not use noise as a treatment but as a parameter to vary."""
    results = {
        "sigma_op": [],
        "cross_cutting": [],
        "assortativity": [],
        "variance": [],
        "n_big": [],
    }

    for i, sigma_op in enumerate(sigma_op_values):
        for r in range(runs):
            seed = seed0 + 1000 * i + r

            m = AgentBasedModel(
                N=200,
                T=10.0,
                dt=0.01,
                alpha=1.0,
                beta=1.0,
                R_sp=0.2,
                R_op=0.12,
                sigma_sp=0.05,
                sigma_op=float(sigma_op),
                seed=seed,
            )
            m.run()

            results["sigma_op"].append(float(sigma_op))
            results["cross_cutting"].append(m.cross_cutting_edge_fraction())
            results["assortativity"].append(m.global_assortativity())
            results["variance"].append(m.opinion_variance())

            labels = hdbscan_cluster_labels_xytheta(
                m,
                theta_scale=theta_scale,
                min_cluster_size=min_cluster_size,
            )
            n_big, _ = count_big_hdbscan_clusters(labels, min_size=min_size_big)
            results["n_big"].append(n_big)

    plot_r5_from_dict(results, savepath=savepath)
