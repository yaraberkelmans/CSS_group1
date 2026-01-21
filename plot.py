import matplotlib.pyplot as plt
import numpy as np


def plot_agents(x, theta, title="Agent Positions and Opinions"):
    """
    Plot the social positions and opinions of agents.
    x: array of shape (N, d) - social positions
    theta: array of shape (N, m) - opinions
    """
    N, d = x.shape
    m = theta.shape[1]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x[:, 0], x[:, 1], c=theta[:, 0], cmap="viridis", s=100)
    plt.colorbar(scatter, label="Opinion Value")

    # inset opinion distributtion histogram at the top right
    ax_inset = plt.gca().inset_axes([0.65, 0.65, 0.3, 0.3])
    ax_inset.hist(theta[:, 0], bins=10, color="gray", edgecolor="black")
    plt.title(title)
    plt.xlabel("Social Position X1")
    plt.ylabel("Social Position X2")
    plt.grid(True)
    plt.show()
