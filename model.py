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
        """
        One Euler–Maruyama update of the system (updates self.agents).
        """
        N = len(self.agents)
        if N == 0:
            return

        d = self.agents[0].d
        m = self.agents[0].m

        dt = self.dt
        alpha = self.alpha
        beta = self.beta
        R_op = self.R_op
        R_sp = self.R_sp
        sigma_op = self.sigma_op
        sigma_sp = self.sigma_sp

        # State at the beginning of the step
        x_state = np.array([a.x for a in self.agents], dtype=float)
        theta_state = np.array([a.theta for a in self.agents], dtype=float)

        dx = np.zeros((N, d))
        dtheta = np.zeros((N, m))

        # Compute drift
        for i in range(N):
            for j in range(N):
                if j == i:
                    continue
                dx[i] += self.agents[i].social_drift(self.agents[j], beta, R_sp)
                dtheta[i] += self.agents[i].opinion_drift(self.agents[j], alpha, R_op)

        # Euler–Maruyama update (noise and timestep)
        dx = (dt / N) * dx + sigma_sp * np.sqrt(dt) * rng.normal(
            size=(N, d), loc=0.0, scale=1.0
        )
        dtheta = (dt / N) * dtheta + sigma_op * np.sqrt(dt) * rng.normal(
            size=(N, m), loc=0.0, scale=1.0
        )

        for i in range(N):
            self.agents[i].x += dx[i]
            self.agents[i].theta += dtheta[i]
            self.agents[i].reflect(low=-0.3, high=0.3)

    def run(self, seed: int = 0, save_every: int = 1) -> tuple[np.ndarray, np.ndarray]:
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
