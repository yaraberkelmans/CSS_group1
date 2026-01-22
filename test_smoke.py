import numpy as np
import pytest

import sim
from agent import Agent
from model import AgentBasedModel


@pytest.fixture
def two_agents_1d():
    """Provide two agents in 1D with deterministic positions and opinions."""
    a0 = Agent(id=0, d=1, m=1)
    a1 = Agent(id=1, d=1, m=1)
    a0.x = np.array([0.0])
    a1.x = np.array([1.0])
    a0.theta = np.array([1.0])
    a1.theta = np.array([1.0])
    return [a0, a1]


@pytest.mark.smoke
def test_social_distance_and_radius_boundary():
    """Smoke: social radius check is inclusive at the boundary."""
    a0 = Agent(id=0, d=2, m=1)
    a1 = Agent(id=1, d=2, m=1)
    a0.x = np.array([0.0, 0.0])
    a1.x = np.array([3.0, 4.0])
    assert a0.social_distance(a1) == 5.0
    assert a0.within_social_radius(a1, radius=5.0) is True
    assert a0.within_social_radius(a1, radius=4.99) is False


@pytest.mark.smoke
def test_social_drift_zero_when_opinions_orthogonal():
    """Smoke: social drift is zero when opinion dot-product is zero."""
    a0 = Agent(id=0, d=1, m=2)
    a1 = Agent(id=1, d=1, m=2)
    a0.x = np.array([0.0])
    a1.x = np.array([0.5])
    a0.theta = np.array([1.0, 0.0])
    a1.theta = np.array([0.0, 1.0])
    drift = a0.social_drift(a1, beta=2.0, R_sp=1.0)
    assert np.allclose(drift, np.zeros_like(a0.x))


@pytest.mark.smoke
@pytest.mark.parametrize("theta_other, expected_sign", [(1.0, 1.0), (-1.0, -1.0)])
def test_social_drift_sign(theta_other, expected_sign):
    """Smoke: social drift sign follows the opinion dot-product."""
    a0 = Agent(id=0, d=1, m=1)
    a1 = Agent(id=1, d=1, m=1)
    a0.x = np.array([0.0])
    a1.x = np.array([1.0])
    a0.theta = np.array([1.0])
    a1.theta = np.array([theta_other])
    drift = a0.social_drift(a1, beta=2.0, R_sp=2.0)
    expected = expected_sign * 2.0 * (a1.x - a0.x)
    assert np.allclose(drift, expected)


@pytest.mark.smoke
def test_opinion_drift_zero_outside_radius():
    """Smoke: opinion drift is zero when agents are outside the opinion radius."""
    a0 = Agent(id=0, d=1, m=1)
    a1 = Agent(id=1, d=1, m=1)
    a0.x = np.array([0.0])
    a1.x = np.array([2.0])
    a0.theta = np.array([0.0])
    a1.theta = np.array([1.0])
    drift = a0.opinion_drift(a1, alpha=3.0, R_op=1.0)
    assert np.allclose(drift, np.zeros_like(a0.theta))


@pytest.mark.smoke
def test_model_step_moves_agents_toward_each_other(two_agents_1d):
    """Smoke: a model step with attraction and no noise moves agents together."""
    model = AgentBasedModel(N=2, d=1, m=1, T=0.1, dt=0.2)
    model.agents = two_agents_1d
    model.alpha = 0.0
    model.beta = 1.0
    model.R_sp = 2.0
    model.R_op = 2.0
    model.sigma_sp = 0.0
    model.sigma_op = 0.0

    rng = np.random.default_rng(0)
    model.step(rng)

    assert np.allclose(model.agents[0].x, np.array([0.1]))
    assert np.allclose(model.agents[1].x, np.array([0.9]))


@pytest.mark.smoke
def test_model_run_records_expected_steps():
    """Smoke: model run records initial state plus every step."""
    model = AgentBasedModel(N=3, d=1, m=1, T=0.02, dt=0.01)
    model.alpha = 0.0
    model.beta = 0.0
    model.sigma_sp = 0.0
    model.sigma_op = 0.0
    x_over_time, theta_over_time = model.run(seed=1, save_every=1)
    assert x_over_time.shape[0] == 3
    assert theta_over_time.shape[0] == 3
    assert x_over_time.shape[1:] == (model.N, model.d)
    assert theta_over_time.shape[1:] == (model.N, model.m)


@pytest.mark.smoke
def test_sim_run_simulation_delegates_to_model_run(monkeypatch):
    """Smoke: sim.run_simulation returns the model.run output."""
    expected_x = np.zeros((2, 1, 1))
    expected_theta = np.ones((2, 1, 1))

    class DummyModel:
        def run(self, seed=0, save_every=1):
            return expected_x, expected_theta

    monkeypatch.setattr(sim, "AgentBasedModel", DummyModel)
    x_over_time, theta_over_time = sim.run_simulation(
        agents=np.array([]),
        T=0.1,
        dt=0.01,
        alpha=1.0,
        beta=1.0,
        R_sp=1.0,
        R_op=1.0,
        sigma_sp=0.0,
        sigma_op=0.0,
        seed=5,
        save_every=2,
    )
    assert np.array_equal(x_over_time, expected_x)
    assert np.array_equal(theta_over_time, expected_theta)


@pytest.mark.smoke
def test_no_interactions_brownian_statistics():
    """Smoke: with alpha=beta=0, states follow Brownian motion statistics."""
    model = AgentBasedModel(N=200, d=1, m=1, T=0.2, dt=0.01)
    model.alpha = 0.0
    model.beta = 0.0
    model.sigma_sp = 0.3
    model.sigma_op = 0.5
    for agent in model.agents:
        agent.x = np.zeros(1)
        agent.theta = np.zeros(1)

    rng = np.random.default_rng(123)
    steps = int(model.T / model.dt)
    for _ in range(steps):
        model.step(rng)

    x_vals = np.array([a.x[0] for a in model.agents])
    theta_vals = np.array([a.theta[0] for a in model.agents])
    expected_var_x = model.sigma_sp**2 * model.T
    expected_var_theta = model.sigma_op**2 * model.T

    mean_tol_x = 4.0 * np.sqrt(expected_var_x / model.N)
    mean_tol_theta = 4.0 * np.sqrt(expected_var_theta / model.N)
    assert abs(np.mean(x_vals)) <= mean_tol_x
    assert abs(np.mean(theta_vals)) <= mean_tol_theta
    assert np.isclose(np.var(x_vals, ddof=1), expected_var_x, rtol=0.3, atol=0.0)
    assert np.isclose(
        np.var(theta_vals, ddof=1), expected_var_theta, rtol=0.3, atol=0.0
    )


@pytest.mark.smoke
def test_no_noise_no_interaction_no_movement():
    """Smoke: with no noise and no interactions, state stays fixed."""
    model = AgentBasedModel(N=2, d=1, m=1, T=0.1, dt=0.05)
    model.alpha = 1.0
    model.beta = 1.0
    model.R_sp = 0.1
    model.R_op = 0.1
    model.sigma_sp = 0.0
    model.sigma_op = 0.0
    model.agents[0].x = np.array([0.0])
    model.agents[1].x = np.array([1.0])
    model.agents[0].theta = np.array([0.2])
    model.agents[1].theta = np.array([-0.4])

    rng = np.random.default_rng(0)
    model.step(rng)

    assert np.allclose(model.agents[0].x, np.array([0.0]))
    assert np.allclose(model.agents[1].x, np.array([1.0]))
    assert np.allclose(model.agents[0].theta, np.array([0.2]))
    assert np.allclose(model.agents[1].theta, np.array([-0.4]))


@pytest.mark.smoke
def test_all_to_all_social_mean_conserved_and_distances_shrink():
    """Smoke: with identical opinions, social positions move to the mean."""
    model = AgentBasedModel(N=4, d=1, m=1, T=0.3, dt=0.1)
    model.alpha = 0.0
    model.beta = 1.0
    model.R_sp = 10.0
    model.R_op = 10.0
    model.sigma_sp = 0.0
    model.sigma_op = 0.0

    positions = np.array([-1.0, -0.5, 0.5, 2.0])
    for i, agent in enumerate(model.agents):
        agent.x = np.array([positions[i]])
        agent.theta = np.array([1.0])

    rng = np.random.default_rng(0)
    mean_initial = np.mean(positions)
    prev_max_dist = np.max(np.abs(positions - mean_initial))

    for _ in range(3):
        model.step(rng)
        x_vals = np.array([a.x[0] for a in model.agents])
        mean_now = np.mean(x_vals)
        max_dist = np.max(np.abs(x_vals - mean_now))
        assert np.isclose(mean_now, mean_initial, atol=1e-12)
        assert max_dist <= prev_max_dist + 1e-12
        prev_max_dist = max_dist


@pytest.mark.smoke
def test_all_to_all_opinion_mean_conserved_and_variance_shrinks():
    """Smoke: with large R_op and no noise, opinions converge to the mean."""
    model = AgentBasedModel(N=4, d=1, m=1, T=0.3, dt=0.1)
    model.alpha = 1.0
    model.beta = 0.0
    model.R_sp = 10.0
    model.R_op = 10.0
    model.sigma_sp = 0.0
    model.sigma_op = 0.0

    opinions = np.array([-1.0, -0.2, 0.4, 1.2])
    for i, agent in enumerate(model.agents):
        agent.x = np.array([0.0])
        agent.theta = np.array([opinions[i]])

    rng = np.random.default_rng(0)
    mean_initial = np.mean(opinions)
    prev_var = np.var(opinions, ddof=0)

    for _ in range(3):
        model.step(rng)
        theta_vals = np.array([a.theta[0] for a in model.agents])
        mean_now = np.mean(theta_vals)
        var_now = np.var(theta_vals, ddof=0)
        assert np.isclose(mean_now, mean_initial, atol=1e-12)
        assert var_now <= prev_var + 1e-12
        prev_var = var_now
