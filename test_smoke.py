import numpy as np
import pytest

from src.agent import Agent
from src.model import AgentBasedModel


@pytest.mark.smoke
def test_agent_distance_and_radius_boundary():
    a0 = Agent(id=0, d=2, m=1)
    a1 = Agent(id=1, d=2, m=1)
    a0.x = np.array([0.0, 0.0])
    a1.x = np.array([3.0, 4.0])
    assert a0.social_distance(a1) == 5.0
    assert a0.within_social_radius(a1, radius=5.0) is True
    assert a0.within_social_radius(a1, radius=4.99) is False


@pytest.mark.smoke
def test_adjacency_matrix_thresholds_and_diagonal():
    model = AgentBasedModel(N=3, d=2, m=1, R_sp=0.2, R_op=0.2, seed=0)
    model.agents[0].x = np.array([0.0, 0.0])
    model.agents[1].x = np.array([0.1, 0.0])
    model.agents[2].x = np.array([0.5, 0.0])

    A_sp, A_op = model.adjacency_matrix()
    assert A_sp.shape == (3, 3)
    assert A_op.shape == (3, 3)
    assert np.all(np.diag(A_sp) == 0)
    assert np.all(np.diag(A_op) == 0)

    assert A_sp[0, 1] == 1
    assert A_sp[1, 0] == 1
    assert A_sp[0, 2] == 0
    assert A_sp[1, 2] == 0


@pytest.mark.smoke
def test_step_social_drift_moves_agents_together():
    model = AgentBasedModel(N=2, d=2, m=1, T=0.2, dt=0.2, seed=0)
    model.alpha = 0.0
    model.beta = 1.0
    model.R_sp = 1.0
    model.R_op = 1.0
    model.sigma_sp = 0.0
    model.sigma_op = 0.0

    model.agents[0].x = np.array([-0.1, 0.0])
    model.agents[1].x = np.array([0.1, 0.0])
    model.agents[0].theta = np.array([1.0])
    model.agents[1].theta = np.array([1.0])

    rng = np.random.default_rng(0)
    model.step(rng)

    assert np.allclose(model.agents[0].x, np.array([-0.08, 0.0]))
    assert np.allclose(model.agents[1].x, np.array([0.08, 0.0]))


@pytest.mark.smoke
def test_step_no_interaction_no_noise_no_movement():
    model = AgentBasedModel(N=2, d=2, m=1, T=0.1, dt=0.05, seed=0)
    model.alpha = 0.0
    model.beta = 0.0
    model.R_sp = 0.1
    model.R_op = 0.1
    model.sigma_sp = 0.0
    model.sigma_op = 0.0

    model.agents[0].x = np.array([0.0, 0.0])
    model.agents[1].x = np.array([0.2, 0.0])
    model.agents[0].theta = np.array([0.2])
    model.agents[1].theta = np.array([-0.4])

    rng = np.random.default_rng(0)
    model.step(rng)

    assert np.allclose(model.agents[0].x, np.array([0.0, 0.0]))
    assert np.allclose(model.agents[1].x, np.array([0.2, 0.0]))
    assert np.allclose(model.agents[0].theta, np.array([0.2]))
    assert np.allclose(model.agents[1].theta, np.array([-0.4]))


@pytest.mark.smoke
def test_run_shapes_with_save_every():
    model = AgentBasedModel(N=3, d=2, m=1, T=0.03, dt=0.01, seed=1)
    model.alpha = 0.0
    model.beta = 0.0
    model.sigma_sp = 0.0
    model.sigma_op = 0.0

    x_over_time, theta_over_time = model.run(save_every=1)
    assert x_over_time.shape[0] == 4
    assert theta_over_time.shape[0] == 4
    assert x_over_time.shape[1:] == (model.N, model.d)
    assert theta_over_time.shape[1:] == (model.N, model.m)


@pytest.mark.smoke
def test_cross_cutting_edge_fraction_no_edges():
    model = AgentBasedModel(N=2, d=2, m=1, R_sp=0.01, seed=0)
    model.agents[0].x = np.array([0.0, 0.0])
    model.agents[1].x = np.array([1.0, 0.0])
    model.agents[0].theta = np.array([1.0])
    model.agents[1].theta = np.array([-1.0])

    assert model.cross_cutting_edge_fraction() == 0.0


@pytest.mark.smoke
def test_mean_spatial_degree_connected_pair():
    model = AgentBasedModel(N=2, d=2, m=1, R_sp=1.0, seed=0)
    model.agents[0].x = np.array([0.0, 0.0])
    model.agents[1].x = np.array([0.5, 0.0])

    assert model.mean_spatial_degree() == 1.0
