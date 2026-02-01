import numpy as np
import matplotlib.pyplot as plt
from src.model import AgentBasedModel
from analysis import hdbscan_cluster_labels_xytheta, count_big_hdbscan_clusters


def run_r5_noise_sweep_dict(
    sigma_op_values,
    runs=1,
    seed0=0,
    theta_scale=1.0,
    min_cluster_size=10,
    min_size_big=10,
):
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

            results["sigma_op"].append(sigma_op)
            results["cross_cutting"].append(m.cross_cutting_edge_fraction())
            results["assortativity"].append(m.global_assortativity())
            results["variance"].append(m.opinion_variance())

            labels = hdbscan_cluster_labels_xytheta(
                m,
                theta_scale=theta_scale,
                min_cluster_size=min_cluster_size,
            )
            n_big, size_by_id = count_big_hdbscan_clusters(labels, min_size=min_size_big)
            results["n_big"].append(n_big)

    return results

def plot_r5_from_dict(results):
    sigma = np.array(results["sigma_op"])
    unique = np.unique(sigma)

    def aggregate(key):
        mean, std = [], []
        for s in unique:
            vals = np.array(results[key])[sigma == s]
            mean.append(vals.mean())
            std.append(vals.std())
        return np.array(mean), np.array(std)

    fig, axs = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

    for ax, key, label in zip(
        axs,
        ["cross_cutting", "assortativity", "variance", "n_big"],
        ["Cross-cutting", "Assortativity", "Opinion variance", "# big clusters"],
    ):
        m, s = aggregate(key)
        ax.plot(unique, m)
        ax.fill_between(unique, m - s, m + s, alpha=0.2)
        ax.set_ylabel(label)

    axs[-1].set_xlabel("sigma_op")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sigmas = np.linspace(0.0, 0.2, 21)
    results = run_r5_noise_sweep_dict(sigmas, runs=20)
    plot_r5_from_dict(results)
