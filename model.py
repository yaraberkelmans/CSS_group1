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

        # create a list of N agents
        self.agents = [Agent(id=i, d=d, m=m) for i in range(N)]

        # initization
        self._initialize_agents()

    def _initialize_agents(self):
        """Set initial social positions and opinions for all agents."""
        for agent in self.agents:
            agent.x = np.random.uniform(
                -0.25, 0.25, size=self.d
            )  # social space [-0.25, 0.25]^2
            agent.theta = np.random.uniform(-1, 1, size=self.m)  # opinion space [-1, 1]

    def step(self, rng):
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
        dx = (dt / N) * dx + sigma_sp * np.sqrt(dt) * rng.normal(size=(N, d))
        dtheta = (dt / N) * dtheta + sigma_op * np.sqrt(dt) * rng.normal(size=(N, m))

        for i in range(N):
            self.agents[i].x += dx[i]
            self.agents[i].theta += dtheta[i]

    def run(self, seed=0, save_every=1):
        """
        Run the simulation and return agent states over time.
        """
        rng = np.random.default_rng(seed)
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
