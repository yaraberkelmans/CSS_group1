import numpy as np
import matplotlib.pyplot as plt
import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import AgentBasedModel

def get_assortativity(params, T=5.0):
    random_seed = np.random.randint(1e6)
    np.random.seed(random_seed)

    model = AgentBasedModel(T=T, seed=random_seed, **params)
    model.run() 

    return model.global_assortativity()

def run_trajectory(params, sample_rate=10):
    model = AgentBasedModel(**params)
    
    times = []
    assortativities = []
    steps = int(model.T / model.dt)
    
    rng = np.random.default_rng()

    # t=0
    times.append(0)
    assortativities.append(model.global_assortativity())
    
    for step in range(1, steps + 1):
        model.step(rng)
        
        if step % sample_rate == 0:
            times.append(step * model.dt)
            assortativities.append(model.global_assortativity())
            
    return np.array(times), np.array(assortativities)

def get_convergence_time(params, target_r=0.9, max_steps=3000):
    model = AgentBasedModel(**params)
    rng = np.random.default_rng()
    
    for step in range(max_steps):
        model.step(rng)
        
        if step % 10 == 0:
            r = model.global_assortativity()
            if r >= target_r:
                return step * model.dt

def fig_5a():
    R_op_values = [0.03, 0.15, 0.3]
    colors = ['red', 'purple', 'green']
    
    base_params = {
        'N': 100, 'T': 10.0, 'dt': 0.01,
        'alpha': 40.0, 'beta': 10.0,
        'R_sp': 0.15,
        'sigma_sp': 0.05, 'sigma_op': 0.05
    }
    
    n_simulations = 10 
    
    plt.figure(figsize=(10, 6))
    
    for idx, r_op in enumerate(R_op_values):
        current_params = base_params.copy()
        current_params['R_op'] = r_op
        
        all_runs = []
        for _ in tqdm.tqdm(range(n_simulations), leave=False):
            times, r_curve = run_trajectory(current_params, sample_rate=20)
            all_runs.append(r_curve)
        
        min_len = min(len(r) for r in all_runs)
        all_runs = np.array([r[:min_len] for r in all_runs])
        times = times[:min_len]
        
        mean_curve = np.mean(all_runs, axis=0)
        std_curve = np.std(all_runs, axis=0)
        
        plt.plot(times, mean_curve, label=f'R_op={r_op}', color=colors[idx], linewidth=2)
        plt.fill_between(times, mean_curve - std_curve, mean_curve + std_curve, 
                         color=colors[idx], alpha=0.2)
        
    plt.title("Fig 5a: Temporal Evolution of Assortativity")
    plt.xlabel("t")
    plt.ylabel("Global Assortativity")
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, 10.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

fig_5a()