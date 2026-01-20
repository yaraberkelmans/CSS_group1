import numpy as np
from agent import Agent

# Here goes the model implementation (i.e., building n agents, defining param space and so on)

class AgentBasedModel:
    """
    The simulation model represent the co-evolving system.
    It manages: the list of agents, simulation parameters, and data storage (for further visualization).
    """
    def __init__(self, 
                 N=100,           # number of agents
                 T=2.5,           # total simulation time
                 dt=0.01,         # time-step size
                 alpha=40.0,      # opinion influence strength
                 beta=10.0,       # social influence strength
                 R_sp=0.15,       # social interaction radius
                 R_op=0.15,       # opinion interaction radius
                 sigma_sp=0.05,   # social noise strength
                 sigma_op=0.05,   # opinion noise strength
                 d=2,             # social space dimension
                 m=1):            # opinion space dimension
        
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
        
        # storage for history
        num_steps = int(T / dt) + 1
        self.history_x = np.zeros((num_steps, N, d))
        self.history_theta = np.zeros((num_steps, N, m))
        
        # initization
        self._initialize_agents()

    def _initialize_agents(self):
        """Set initial social positions and opinions for all agents."""
        for agent in self.agents:
            agent.x = np.random.uniform(-0.25, 0.25, size=self.d)   # social space [-0.25, 0.25]^2
            agent.theta = np.random.uniform(-1, 1, size=self.m)     # opinion space [-1, 1]

    def compute_drifts(self):
        """
        Computes the deterministic drift components for all agents based on their current state.

        Returns:
            drift_x_all: The total social force vector for each agent.
            drift_theta_all: The total opinion influence vector for each agent.
        """
        # initialize arrays
        drift_x_all = np.zeros((self.N, self.d))
        drift_theta_all = np.zeros((self.N, self.m))

        # iterate over all pairs of agents to compute interactions
        for i, agent_i in enumerate(self.agents):
            sum_drift_x = np.zeros(self.d)
            sum_drift_theta = np.zeros(self.m)

            for j, agent_j in enumerate(self.agents):
                if i == j: 
                    continue
                
                # social force (U) exerted by agent_j on agent_i
                u_ij = agent_i.social_drift(agent_j, self.beta, self.R_sp)
                
                # opinion influence (V) exerted by agent_j on agent_i
                v_ij = agent_i.opinion_drift(agent_j, self.alpha, self.R_op)
                
                # accumulate the interactions
                sum_drift_x += u_ij
                sum_drift_theta += v_ij

            # divide by N
            drift_x_all[i] = sum_drift_x / self.N
            drift_theta_all[i] = sum_drift_theta / self.N
            
        return drift_x_all, drift_theta_all
