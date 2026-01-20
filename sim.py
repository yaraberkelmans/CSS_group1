import numpy as np
from agent import Agent

def euler_step(agents, dt, alpha, beta, R_op, R_sp, sigma_op, sigma_sp, rng):
    """
    Perform one Euler–Maruyama update of the agent system.
    """
    N = len(agents)
    if N == 0:
        return

    d = agents[0].d
    m = agents[0].m

    # State at the beginning of the step
    x_state = np.array([a.x for a in agents], dtype=float)
    theta_state = np.array([a.theta for a in agents], dtype=float)

    dx = np.zeros((N, d))
    dtheta = np.zeros((N, m))

    # Compute interaction effects
    state = []
    for i in range(N):
        a = Agent(agents[i].id, d, m)
        a.x = x_state[i].copy()
        a.theta = theta_state[i].copy()
        state.append(a)

    # Compute drift 
    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            dx[i] += state[i].social_drift(state[j], beta, R_sp)
            dtheta[i] += state[i].opinion_drift(state[j], alpha, R_op)

    # Euler–Maruyama update (noise and timestep)
    dx = (dt / N) * dx + sigma_sp * np.sqrt(dt) * rng.normal(size=(N, d))
    dtheta = (dt / N) * dtheta + sigma_op * np.sqrt(dt) * rng.normal(size=(N, m))

    for i in range(N):
        agents[i].x = x_state[i] + dx[i]
        agents[i].theta = theta_state[i] + dtheta[i]


def run_simulation(agents, T, dt, alpha, beta, R_op, R_sp, sigma_op, sigma_sp, seed=0, save_every=1):
    """
    Run the simulation, record and return agent states over time.
    """
    rng = np.random.default_rng(seed)
    steps = int(T / dt)
    x_over_time = []
    theta_over_time = []

    # Record initial state
    x_over_time.append(np.array([a.x for a in agents], dtype=float))
    theta_over_time.append(np.array([a.theta for a in agents], dtype=float))

    for step in range(1, steps + 1):
        euler_step(agents, dt, alpha, beta, R_op, R_sp, sigma_op, sigma_sp, rng)

        # Record agent states at regular intervals
        if step % save_every == 0:
            x_over_time.append(np.array([a.x for a in agents], dtype=float))
            theta_over_time.append(np.array([a.theta for a in agents], dtype=float))

    return np.array(x_over_time), np.array(theta_over_time)

