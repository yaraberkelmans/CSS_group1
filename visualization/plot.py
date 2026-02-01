import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import AgentBasedModel


def plot_snapshot(model: "AgentBasedModel", title: str, savepath: str = None) -> None:

    # get last theta values
    final_x = np.array([a.x for a in model.agents])
    final_theta = np.array([a.theta[0] for a in model.agents])

    fig, ax = plt.subplots(figsize=(7, 7))

    # main scatter plot
    sc = ax.scatter(
        final_x[:, 0],
        final_x[:, 1],
        c=final_theta,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        edgecolor="k",
        s=60,
        alpha=0.8,
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Opinion (Theta)")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Social Space X")
    ax.set_ylabel("Social Space Y")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.grid(True, linestyle="--", alpha=0.3)

    # inset axes for opinion histogram
    ax_inset = ax.inset_axes([0.65, 0.75, 0.35, 0.25])
    ax_inset.hist(final_theta, bins=15, range=(-0.5, 0.5), color="blue", alpha=0.8)

    ax_inset.set_xlim(-0.6, 0.6)
    ax_inset.set_xticks([-0.5, 0.0, 0.5])
    ax_inset.set_yticks([])
    ax_inset.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, dpi=300)


def plot_snapshot_with_edges(
    model: "AgentBasedModel", title: str, savepath: str = None
) -> None:
    """
    plot agents + network edges from adjacency matrix.
    """
    final_x = np.array([a.x for a in model.agents])
    final_theta = np.array([a.theta[0] for a in model.agents])

    A_sp, A_op = model.adjacency_matrix()

    fig, ax = plt.subplots(figsize=(7, 7))

    # draw edges
    N = model.N
    for i in range(N):
        for j in range(i + 1, N):
            if A_sp[i, j] == 1:
                ax.plot(
                    [final_x[i, 0], final_x[j, 0]],
                    [final_x[i, 1], final_x[j, 1]],
                    linewidth=0.6,
                    alpha=0.4,
                    color="gray",
                )

    # draw nodes
    sc = ax.scatter(
        final_x[:, 0],
        final_x[:, 1],
        c=final_theta,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        edgecolor="k",
        s=60,
        alpha=0.85,
        zorder=3,
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Opinion (Theta)")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Social Space X")
    ax.set_ylabel("Social Space Y")

    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, dpi=300)


def plot_polarisation_vs_Rop(
    polarisation_values: np.ndarray,
    savepath: str = None,
) -> None:
    """Creates multiple plots of cross_cutting_fraction, global_assortativity, opinion_variance vs R_op values"""

    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    # polarisation values shape: (runs, len(R_op_values), 5)
    polarisation_values_mean = np.mean(polarisation_values, axis=0)
    R_op_values = polarisation_values_mean[:, 0]

    # variance across runs can also be plotted if desired
    polarisation_std = np.std(polarisation_values, axis=0)

    # data
    axs[0].plot(R_op_values, polarisation_values_mean[:, 1], color="blue")
    # plot std as shaded area
    axs[0].fill_between(
        R_op_values,
        polarisation_values_mean[:, 1] - polarisation_std[:, 1],
        polarisation_values_mean[:, 1] + polarisation_std[:, 1],
        color="blue",
        alpha=0.2,
    )
    axs[0].set_title("Cross-Cutting Edge Fraction vs R_op")
    axs[0].set_xlabel("R_op")
    axs[0].set_ylabel("Cross-Cutting Edge Fraction")
    axs[0].grid(True, linestyle="--", alpha=0.3)

    axs[1].plot(R_op_values, polarisation_values_mean[:, 2], color="green")
    axs[1].fill_between(
        R_op_values,
        polarisation_values_mean[:, 2] - polarisation_std[:, 2],
        polarisation_values_mean[:, 2] + polarisation_std[:, 2],
        color="green",
        alpha=0.2,
    )
    axs[1].set_title("Global Assortativity vs R_op")
    axs[1].set_xlabel("R_op")
    axs[1].set_ylabel("Global Assortativity")
    axs[1].grid(True, linestyle="--", alpha=0.3)

    axs[2].plot(R_op_values, polarisation_values_mean[:, 3], color="red")
    axs[2].fill_between(
        R_op_values,
        polarisation_values_mean[:, 3] - polarisation_std[:, 3],
        polarisation_values_mean[:, 3] + polarisation_std[:, 3],
        color="red",
        alpha=0.2,
    )
    axs[2].set_title("Opinion Variance vs R_op")
    axs[2].set_xlabel("R_op")
    axs[2].set_ylabel("Opinion Variance")
    axs[2].grid(True, linestyle="--", alpha=0.3)

    axs[3].plot(R_op_values, polarisation_values_mean[:, 4].astype(int), color="purple")
    axs[3].fill_between(
        R_op_values,
        polarisation_values_mean[:, 4].astype(int) - polarisation_std[:, 4],
        polarisation_values_mean[:, 4].astype(int) + polarisation_std[:, 4],
        color="purple",
        alpha=0.2,
    )
    axs[3].set_title("# Large Clusters vs R_op")
    axs[3].set_xlabel("R_op")
    axs[3].set_ylabel("# Large Clusters (>=10)")
    axs[3].grid(True, linestyle="--", alpha=0.3)
    axs[3].set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, dpi=300)


def plot_depolarisation_vs_noise(
    experiment_results: np.ndarray,
    noise_range: np.ndarray = np.arange(0.01, 0.11, 0.01),
    savepath: str = None,
) -> None:
    """Plot noise depolarisation experiment results."""

    print(experiment_results.shape)
    pre_treatment_assortativity = experiment_results[:, 0, :, 0]
    pre_treatment_opinion_variance = experiment_results[:, 0, :, 1]
    post_treatment_assortativity = experiment_results[:, 1, :, 0]
    post_treatment_opinion_variance = experiment_results[:, 1, :, 1]

    mean_pre_assort = np.mean(pre_treatment_assortativity, axis=0)
    mean_post_assort = np.mean(post_treatment_assortativity, axis=0)
    mean_pre_opvar = np.mean(pre_treatment_opinion_variance, axis=0)
    mean_post_opvar = np.mean(post_treatment_opinion_variance, axis=0)

    std_var_post_opvar = np.std(post_treatment_opinion_variance, axis=0)
    std_var_post_assort = np.std(post_treatment_assortativity, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        noise_range,
        mean_pre_assort,
        label="Assortativity",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        noise_range,
        mean_post_assort,
        color="blue",
    )
    ax.fill_between(
        noise_range,
        mean_post_assort - std_var_post_assort,
        mean_post_assort + std_var_post_assort,
        color="blue",
        alpha=0.2,
    )

    ax.set_xlabel("Depolarisation Noise (σ_op = σ_sp)")
    ax.set_ylabel("Global Assortativity", color="darkblue")

    # second y axis for opinion variance
    ax2 = ax.twinx()
    ax2.plot(
        noise_range,
        mean_pre_opvar,
        label="Opinion Variance",
        color="orange",
        linestyle="--",
    )
    ax2.plot(
        noise_range,
        mean_post_opvar,
        color="orange",
    )
    ax2.fill_between(
        noise_range,
        mean_post_opvar - std_var_post_opvar,
        mean_post_opvar + std_var_post_opvar,
        color="orange",
        alpha=0.2,
    )
    ax2.set_ylabel("Opinion Variance", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax.set_title("Global Assortativity vs Depolarisation Noise")
    ax.grid(True, linestyle="--", alpha=0.3)

    # legends for both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # legend that shows dashed pre treatment, solid post treatment

    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, dpi=300)


def plot_depolarisation_vs_edge_removal(
    experiment_results: np.ndarray,
    ignore_fraction_range: np.ndarray = np.arange(0.0, 0.55, 0.05),
    savepath: str = None,
) -> None:
    """Plot edge removal depolarisation experiment results."""

    print(experiment_results.shape)
    pre_treatment_assortativity = experiment_results[:, 0, :, 0]
    pre_treatment_opinion_variance = experiment_results[:, 0, :, 1]
    post_treatment_assortativity = experiment_results[:, 1, :, 0]

    post_treatment_opinion_variance = experiment_results[:, 1, :, 1]
    mean_pre_assort = np.mean(pre_treatment_assortativity, axis=0)
    mean_post_assort = np.mean(post_treatment_assortativity, axis=0)
    mean_pre_opvar = np.mean(pre_treatment_opinion_variance, axis=0)
    mean_post_opvar = np.mean(post_treatment_opinion_variance, axis=0)
    std_var_post_opvar = np.std(post_treatment_opinion_variance, axis=0)
    std_var_post_assort = np.std(post_treatment_assortativity, axis=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        ignore_fraction_range,
        mean_pre_assort,
        label="Assortativity",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        ignore_fraction_range,
        mean_post_assort,
        color="blue",
    )
    ax.fill_between(
        ignore_fraction_range,
        mean_post_assort - std_var_post_assort,
        mean_post_assort + std_var_post_assort,
        color="blue",
        alpha=0.2,
    )
    ax.set_xlabel("Fraction of Ignored Edges")
    ax.set_ylabel("Global Assortativity", color="darkblue")
    # second y axis for opinion variance
    ax2 = ax.twinx()
    ax2.plot(
        ignore_fraction_range,
        mean_pre_opvar,
        label="Opinion Variance",
        color="orange",
        linestyle="--",
    )
    ax2.plot(
        ignore_fraction_range,
        mean_post_opvar,
        color="orange",
    )
    ax2.fill_between(
        ignore_fraction_range,
        mean_post_opvar - std_var_post_opvar,
        mean_post_opvar + std_var_post_opvar,
        color="orange",
        alpha=0.2,
    )
    ax2.set_ylabel("Opinion Variance", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax.set_title("Global Assortativity vs Ignored Edge Fraction")
    ax.grid(True, linestyle="--", alpha=0.3)
    # legends for both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # legend that shows dashed pre treatment, solid post treatment
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    plt.tight_layout()
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, dpi=300)


def plot_depolarisation_vs_edge_removal_individual_trajectories(
    experiment_results: np.ndarray,
    ignore_fraction_range: np.ndarray = np.arange(0.0, 0.55, 0.05),
    savepath: str = None,
) -> None:
    """Plot edge removal depolarisation experiment results. Each trajecotry should have a different color."""

    pre_treatment_assortativity = experiment_results[:, 0, :, 0]
    pre_treatment_opinion_variance = experiment_results[:, 0, :, 1]
    post_treatment_assortativity = experiment_results[:, 1, :, 0]
    post_treatment_opinion_variance = experiment_results[:, 1, :, 1]

    mean_pre_assort = np.mean(pre_treatment_assortativity, axis=0)
    mean_post_assort = np.mean(post_treatment_assortativity, axis=0)
    mean_pre_opvar = np.mean(pre_treatment_opinion_variance, axis=0)
    mean_post_opvar = np.mean(post_treatment_opinion_variance, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        ignore_fraction_range,
        mean_pre_assort,
        label="Assortativity",
        color="blue",
        linestyle="--",
    )
    ax.plot(
        ignore_fraction_range,
        mean_post_assort,
        color="blue",
    )
    # each trajectory with different color
    for i, pre_assort in enumerate(pre_treatment_assortativity):
        # colours from a colormap
        color = plt.cm.viridis(i / len(pre_treatment_assortativity))

        ax.plot(
            ignore_fraction_range,
            pre_assort,
            color=color,
            alpha=0.3,
            ls="--",
        )

    for i, post_assort in enumerate(post_treatment_assortativity):
        color = plt.cm.viridis(i / len(post_treatment_assortativity))
        ax.plot(
            ignore_fraction_range,
            post_assort,
            color=color,
            alpha=0.3,
        )

    ax.set_xlabel("Fraction of Ignored Edges")
    ax.set_ylabel("Global Assortativity", color="darkblue")
    # second y axis for opinion variance
    ax2 = ax.twinx()
    ax2.plot(
        ignore_fraction_range,
        mean_pre_opvar,
        label="Opinion Variance",
        color="orange",
        linestyle="--",
    )
    ax2.plot(
        ignore_fraction_range,
        mean_post_opvar,
        color="orange",
    )

    # each trajectory with different color
    for _, pre_opvar in enumerate(pre_treatment_opinion_variance):

        # color from cmap
        color = plt.cm.plasma(_, len(pre_treatment_opinion_variance))
        ax2.plot(
            ignore_fraction_range,
            pre_opvar,
            color=color,
            ls="--",
            alpha=0.3,
        )

    for _, post_opvar in enumerate(post_treatment_opinion_variance):
        color = plt.cm.plasma(_, len(post_treatment_opinion_variance))
        ax2.plot(
            ignore_fraction_range,
            post_opvar,
            color=color,
            alpha=0.3,
        )

    ax2.set_ylabel("Opinion Variance", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax.set_title("Global Assortativity vs Ignored Edge Fraction")
    ax.grid(True, linestyle="--", alpha=0.3)
    # legends for both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # legend that shows dashed pre treatment, solid post treatment
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    plt.tight_layout()
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, dpi=300)


def plot_seed_dependence(
    result: np.ndarray, seeds: np.ndarray, savepath: str = None
) -> None:
    """
    Plot distributions of final global assortativity and opinion variance across seeds.
    """
    final_assortativities = result[:, 0]
    final_opinion_variances = result[:, 1]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].hist(
        final_assortativities,
        bins=15,
        color="lightblue",
        edgecolor="darkblue",
        alpha=0.85,
    )
    axs[0].set_title("Assortativity")
    axs[0].set_xlabel("Global Assortativity")
    axs[0].set_ylabel("Count")

    axs[1].hist(
        final_opinion_variances,
        bins=15,
        color="lightcoral",
        edgecolor="darkred",
        alpha=0.85,
    )
    axs[1].set_title("Opinion Variance")
    axs[1].set_xlabel("Opinion Variance")
    axs[1].set_ylabel("Count")

    fig.suptitle(
        r"Distribution of Final States"
        "\n"
        r"($N=100, \alpha=40, \beta=10, \sigma=0.05$)",
        fontsize=16,
    )
    fig.tight_layout()
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, dpi=300)
