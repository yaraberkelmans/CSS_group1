from model import AgentBasedModel
from plot import plot_agents

import numpy as np


def run_simulation():
    model = AgentBasedModel()
    x_over_time, theta_over_time = model.run(seed=0, save_every=10)

    return x_over_time, theta_over_time


if __name__ == "__main__":

    iterations = 2
    xss = []
    thetass = []
    for i in range(iterations):
        xs, thetas = run_simulation()
        xss.append(xs[-1])
        thetass.append(thetas[-1])

    xs = np.mean(xss, axis=0)
    thetas = np.mean(thetass, axis=0)

    plot_agents(xs, thetas, title="Final Agent Positions and Opinions")
