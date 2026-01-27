import numpy as np
import matplotlib.pyplot as plt


def opinion_entropy(theta, bins: int = 30, value_range=(-1.0, 1.0), normalize: bool = True) -> float:
    """
    Computes the Shannon entropy of the opinion distribution.
    The opinions are binned over a fixed range and the Shannon entropy of the
    resulting discrete distribution is calculated. If normalize is True, the
    entropy is normalized by the maximum possible entropy so that the output
    lies in the interval [0, 1].
    Returns the (normalized) Shannon entropy of the opinion distribution.
    """
    th = np.asarray(theta, dtype=float).reshape(-1)

    counts, _ = np.histogram(th, bins=bins, range=value_range)
    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts / total
    p = p[p > 0]

    H = -np.sum(p * np.log(p))
    if normalize:
        H_max = np.log(bins)
        if H_max > 0:
            H = H / H_max

    return float(H)


def get_square_adjacency(A) -> np.ndarray:
    """
    Wrapper function that ensures a single square (N × N) adjacency matrix is used.
    The model may return either a single adjacency matrix with shape (N, N) or a
    stacked array with shape (2, N, N). In the stacked case, the first adjacency
    matrix is selected. If the input cannot be interpreted as a valid square
    adjacency matrix, an error is raised.
    Returns the correct shape adjencency matrix. 
    """
    A = np.asarray(A)

    if A.ndim == 3 and A.shape[0] == 2 and A.shape[1] == A.shape[2]:
        A = A[0]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"Expected an (N, N) adjacency matrix or a stacked (2, N, N) array, "
            f"but got shape {A.shape}."
        )

    return A


def connectivity_metrics(A) -> dict:
    """
    Computes connectivity metrics from an adjacency matrix.
    Returns  a dictionary containing: number of connected components, 
    size of the largest connected component,
    average node degree and an array of node degrees. 
    """
    A = get_square_adjacency(A)
    N = A.shape[0]

    degrees = A.sum(axis=1)
    avg_degree = float(degrees.mean()) if N > 0 else 0.0

    visited = np.zeros(N, dtype=bool)
    comp_sizes = []

    for start in range(N):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        size = 0

        while stack:
            v = stack.pop()
            size += 1
            neighbors = np.where(A[v] > 0)[0]
            for u in neighbors:
                if not visited[u]:
                    visited[u] = True
                    stack.append(u)

        comp_sizes.append(size)

    return {
        "n_components": int(len(comp_sizes)),
        "largest_component": int(max(comp_sizes)) if comp_sizes else 0,
        "avg_degree": avg_degree,
        "degrees": degrees,
    }


def clustering_coefficients(A) -> dict:
    """
    Computes clustering coefficients.
    Local clustering coefficients are computed for each node, together with
    their average and the global clustering coefficient (transitivity).
    Returns a dictionary containing clustering coefficients on different scales. 
    """
    A = get_square_adjacency(A).astype(int)
    N = A.shape[0]
    if N == 0:
        return {"C_local": np.array([]), "C_avg": 0.0, "C_global": 0.0}

    k = A.sum(axis=1).astype(float)

    A2 = A @ A
    A3 = A2 @ A
    triangles_per_node = np.diag(A3) / 2.0
    total_triangles = triangles_per_node.sum() / 3.0

    denom = k * (k - 1) / 2.0
    C_local = np.zeros(N, dtype=float)
    mask = denom > 0
    C_local[mask] = triangles_per_node[mask] / denom[mask]
    C_avg = float(C_local.mean())

    connected_triples = float(np.sum(k * (k - 1) / 2.0))
    C_global = float((3.0 * total_triangles) / connected_triples) if connected_triples > 0 else 0.0

    return {"C_local": C_local, "C_avg": C_avg, "C_global": C_global}


def metrics_over_time(model, bins: int = 30, seed: int | None = None):
    """
    Computes opinion entropy, connectivity metrics, and clustering coefficients
    at every simulation time step.
    Returns a dictionary containing all metrics measurements over time . 
    """
    rng = np.random.default_rng(seed)
    n_steps = int(model.T / model.dt)

    t = np.zeros(n_steps)
    entropy = np.zeros(n_steps)
    n_components = np.zeros(n_steps, dtype=int)
    largest_component = np.zeros(n_steps, dtype=int)
    avg_degree = np.zeros(n_steps)
    C_avg = np.zeros(n_steps)
    C_global = np.zeros(n_steps)

    for step in range(n_steps):
        model.step(rng)

        theta = np.array([a.theta[0] for a in model.agents], dtype=float)
        entropy[step] = opinion_entropy(theta, bins=bins)

        A = model.adjacency_matrix()
        conn = connectivity_metrics(A)
        clust = clustering_coefficients(A)

        t[step] = (step + 1) * model.dt
        n_components[step] = conn["n_components"]
        largest_component[step] = conn["largest_component"]
        avg_degree[step] = conn["avg_degree"]
        C_avg[step] = clust["C_avg"]
        C_global[step] = clust["C_global"]

    return {
        "t": t,
        "entropy": entropy,
        "n_components": n_components,
        "largest_component": largest_component,
        "avg_degree": avg_degree,
        "C_avg": C_avg,
        "C_global": C_global,
    }

def plot_metrics_over_time(out: dict):
    """
    Plots all metrics over time.
    """
    t = out["t"]

    plt.figure(figsize=(6, 4))
    plt.plot(t, out["entropy"], linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Normalized opinion entropy")
    plt.title("Entropy over time")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(t, out["n_components"], linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("# connected components")
    plt.title("Connectivity over time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(t, out["C_avg"], linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Average clustering coefficient")
    plt.title("Clustering coefficient over time")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from model import AgentBasedModel

    init_seed = 0
    dyn_seed = 0

    np.random.seed(init_seed)

    model = AgentBasedModel(
        N=100,
        T=2.5,
        dt=0.01,
        alpha=40.0,
        beta=10.0,
        R_sp=0.15,
        R_op=0.15,
        sigma_sp=0.05,
        sigma_op=0.05,
    )

    out = metrics_over_time(model, bins=30, seed=dyn_seed)
    plot_metrics_over_time(out)