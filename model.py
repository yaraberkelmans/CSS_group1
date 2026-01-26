import numpy as np
import tqdm
from agent import Agent

# Here goes the model implementation (i.e., building n agents, defining param space and so on)


class AgentBasedModel:
    """
    The simulation model represent the co-evolving system.
    It manages: the list of agents, simulation parameters, and data storage (for further visualization).
    """

    def __init__(
        self,
        N=100,  # number of agents
        T=2.5,  # total simulation time
        dt=0.01,  # time-step size
        alpha=40.0,  # opinion influence strength
        beta=10.0,  # social influence strength
        R_sp=0.15,  # social interaction radius
        R_op=0.15,  # opinion interaction radius
        sigma_sp=0.05,  # social noise strength
        sigma_op=0.05,  # opinion noise strength
        d=2,  # social space dimension
        m=1,
        seed=0,
    ):  # opinion space dimension

        self.N = N
        self.T = T
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.R_sp = R_sp
        self.R_op = R_op
        self.sigma_sp = sigma_sp
        self.sigma_op = sigma_op
        self.d = d
        self.m = m
        self.seed = seed

        # create a list of N agents
        self.agents = [Agent(id=i, d=d, m=m) for i in range(N)]

        # initization
        self._initialize_agents()

    def _initialize_agents(self):
        """Set initial social positions and opinions for all agents."""
        rng = np.random.default_rng(self.seed)
        for agent in self.agents:
            agent.x = rng.uniform(
                -0.25, 0.25, size=self.d
            )  # social space [-0.25, 0.25]^2
            agent.theta = rng.uniform(-1, 1, size=self.m)  # opinion space [-1, 1]

    def step(self, rng: np.random.Generator) -> None:
        """Performs a single time step update for all agents."""

        N = self.N
        if N == 0:
            return

        # copy to local variable for quicker access
        dt = self.dt
        alpha = self.alpha
        beta = self.beta
        R_op2 = self.R_op * self.R_op
        R_sp2 = self.R_sp * self.R_sp
        sigma_op = self.sigma_op
        sigma_sp = self.sigma_sp

        # stack state into arrays
        X = np.array([a.x for a in self.agents], dtype=float)  # (N, d)
        Theta = np.array([a.theta for a in self.agents], dtype=float)  # (N, m)

        # adjacency (float so we can multiply)
        A_sp, A_op = self.adjacency_matrix()
        A_sp = A_sp.astype(float)
        A_op = A_op.astype(float)

        # social opinion sign
        dots = Theta @ Theta.T  # (N, N)
        S = np.sign(dots)
        S[np.abs(dots) < 1e-10] = 0.0

        # pairwise differences
        dX = X[None, :, :] - X[:, None, :]  # (N, N, d) = x_j - x_i
        dTheta = Theta[None, :, :] - Theta[:, None, :]  # (N, N, m)

        # drifts
        drift_x = beta * ((A_sp * S)[:, :, None] * dX).sum(axis=1)  # (N, d)
        drift_theta = alpha * (A_op[:, :, None] * dTheta).sum(axis=1)  # (N, m)

        # Euler–Maruyama update
        dx = (dt / N) * drift_x + sigma_sp * np.sqrt(dt) * rng.normal(size=X.shape)
        dtheta = (dt / N) * drift_theta + sigma_op * np.sqrt(dt) * rng.normal(
            size=Theta.shape
        )

        # update agents
        for i, a in enumerate(self.agents):
            a.x = X[i] + dx[i]
            a.theta = Theta[i] + dtheta[i]
            # reflect at boundaries
            a.reflect(low=-0.3, high=0.3)

    def run(self, save_every: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the simulation and return agent states over time.
        """
        rng = np.random.default_rng(self.seed)
        steps = int(self.T / self.dt)

        x_over_time = []
        theta_over_time = []

        # Record initial state
        x_over_time.append(np.array([a.x for a in self.agents], dtype=float))
        theta_over_time.append(np.array([a.theta for a in self.agents], dtype=float))

        for step in tqdm.tqdm(range(1, steps + 1)):
            self.step(rng)

            # Record agent states at regular intervals
            if step % save_every == 0:
                x_over_time.append(np.array([a.x for a in self.agents], dtype=float))
                theta_over_time.append(
                    np.array([a.theta for a in self.agents], dtype=float)
                )

        return np.array(x_over_time), np.array(theta_over_time)

    def adjacency_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates adjacency matrices for social and opinion spaces
        """

        # stack agent positions
        X = np.array([agent.x for agent in self.agents])
        # pairwse squared distances of shape (N, N, d)
        D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)

        # create adjacency matrices based on distance thresholds
        A_sp = (D2 <= self.R_sp**2).astype(int)
        A_op = (D2 <= self.R_op**2).astype(int)

        # self-distance is always zero
        np.fill_diagonal(A_sp, 0)
        np.fill_diagonal(A_op, 0)

        return A_sp, A_op

    def global_assortativity(self) -> float:
        """
        Calculates the global assortativity coefficient 'r'.
        """
        # get the adjacency matrix
        adj_matrix_social, adj_matrix_opinion = self.adjacency_matrix()

        # calculate degrees
        degrees = np.sum(adj_matrix_social, axis=1)
        total_degree = np.sum(degrees)

        thetas = np.array([agent.theta[0] for agent in self.agents])
        theta_bar = np.sum(degrees * thetas) / total_degree

        denominator = np.sum(degrees * (thetas - theta_bar) ** 2)

        row, col = np.where(adj_matrix_social == 1)
        diffs = thetas - theta_bar
        numerator = np.sum(diffs[row] * diffs[col])

        return numerator / denominator if denominator != 0 else 0.0

    def cross_cutting_edge_fraction(
        self, eps: float = 1e-10, exclude_neutral: bool = False
    ) -> float:
        """Calculates the fraction of cross-cutting edges in the social network."""
        A_sp, _ = self.adjacency_matrix()
        thetas = np.array([agent.theta[0] for agent in self.agents], dtype=float)

        # consider each undirected edge once
        iu = np.triu_indices_from(A_sp, k=1)
        edges = A_sp[iu] == 1
        if not edges.any():
            return 0.0

        ti = thetas[iu[0]]
        tj = thetas[iu[1]]

        if exclude_neutral:
            neutral = (np.abs(ti) < eps) | (np.abs(tj) < eps)
            edges = edges & (~neutral)
            if not edges.any():
                return 0.0
            ti = ti[edges]
            tj = tj[edges]
            return np.mean(ti * tj < 0)

        # include neutral in denominator, but they won't count as cross-cutting
        return np.mean((ti[edges] * tj[edges]) < 0)

    def mean_spatial_degree(self) -> float:
        """Calculates the mean spatial degree of the social network."""
        A_sp, _ = self.adjacency_matrix()
        return A_sp.sum() / self.N

    def opinion_variance(self) -> float:
        """Calculates the variance of opinions among agents."""
        thetas = np.array([agent.theta[0] for agent in self.agents], dtype=float)
        return np.var(thetas)
