import matplotlib.pyplot as plt
import numpy as np

from model import AgentBasedModel
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
