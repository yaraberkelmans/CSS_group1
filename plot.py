import matplotlib.pyplot as plt
import numpy as np

from model import AgentBasedModel


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

    R_op_values = polarisation_values[:, 0]

    axs[0].plot(R_op_values, polarisation_values[:, 1], marker="o", color="blue")
    axs[0].set_title("Cross-Cutting Edge Fraction vs R_op")
    axs[0].set_xlabel("R_op")
    axs[0].set_ylabel("Cross-Cutting Edge Fraction")
    axs[0].grid(True, linestyle="--", alpha=0.3)

    axs[1].plot(R_op_values, polarisation_values[:, 2], marker="o", color="green")
    axs[1].set_title("Global Assortativity vs R_op")
    axs[1].set_xlabel("R_op")
    axs[1].set_ylabel("Global Assortativity")
    axs[1].grid(True, linestyle="--", alpha=0.3)

    axs[2].plot(R_op_values, polarisation_values[:, 3], marker="o", color="red")
    axs[2].set_title("Opinion Variance vs R_op")
    axs[2].set_xlabel("R_op")
    axs[2].set_ylabel("Opinion Variance")
    axs[2].grid(True, linestyle="--", alpha=0.3)

    axs[3].plot(R_op_values, polarisation_values[:, 4], marker="o", color="purple")
    axs[3].set_title("Number of Large Clusters vs R_op")
    axs[3].set_xlabel("R_op")
    axs[3].set_ylabel("Number of Large Clusters (size >= 10)")
    axs[3].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, dpi=300)
