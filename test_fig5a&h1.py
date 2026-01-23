import numpy as np
import matplotlib.pyplot as plt
import tqdm
from model import AgentBasedModel

# reproducing fig 5a, testing h1

def get_assortativity(params, T=5.0):
    model = AgentBasedModel(T=T, **params)
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

# ==========================================
# 1. Figure 5a
# ==========================================
def fig_5a():
    print("\n" + "="*60)
    print(">>> 1. Figure 5a: Temporal evolution (varying R_op)")
    print("="*60)
    
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
        print(f"Simulating R_op = {r_op} ...")
        
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


# ==========================================
# 2. test H1: speed of emergence varying alpha
# ==========================================
def test_h1():
    print("\n" + "="*60)
    print(">>> 2. Testing H1: Speed of emergence (varying Alpha)")
    print("="*60)
    
    alpha_values = [5.0, 40.0, 80.0]
    colors = ['blue', 'orange', 'red']
    
    base_params = {
        'N': 100, 'T': 10.0, 'dt': 0.01,
        'beta': 10.0,       
        'R_sp': 0.15, 
        'R_op': 0.15,
        'sigma_sp': 0.05, 'sigma_op': 0.05
    }
    
    n_simulations = 10
    
    plt.figure(figsize=(10, 6))
    
    for idx, alpha in enumerate(alpha_values):
        print(f"Simulating Alpha = {alpha} ...")
        
        current_params = base_params.copy()
        current_params['alpha'] = alpha
        
        all_runs = []
        for _ in tqdm.tqdm(range(n_simulations), leave=False):
            times, r_curve = run_trajectory(current_params, sample_rate=20)
            all_runs.append(r_curve)
            
        min_len = min(len(r) for r in all_runs)
        all_runs = np.array([r[:min_len] for r in all_runs])
        times = times[:min_len]
        
        mean_curve = np.mean(all_runs, axis=0)
        std_curve = np.std(all_runs, axis=0)
        
        label_txt = f'Alpha={alpha} (Strong)' if alpha == 80.0 else f'Alpha={alpha}'
        plt.plot(times, mean_curve, label=label_txt, color=colors[idx], linewidth=2)
        plt.fill_between(times, mean_curve - std_curve, mean_curve + std_curve, 
                         color=colors[idx], alpha=0.2)
        
    plt.title("H1 Test: Impact of Opinion Strength (alpha) on convergence speed")
    plt.xlabel("t")
    plt.ylabel("Global Assortativity")
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, 10.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    fig_5a()
    #test_h1()