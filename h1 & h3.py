import numpy as np
import matplotlib.pyplot as plt
import tqdm
from model import AgentBasedModel
from scipy.stats import skew, kurtosis
import os


# reproducing fig 5a, testing h1, and testing h3

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
    print(">>> 2. Comprehensive H1 Test: Trajectory, Speed, and Ratio")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # plot A
    
    print("Generating Part A")
    alpha_values_traj = [5.0, 40.0, 80.0]
    colors_traj = ['blue', 'orange', 'red']
    
    base_params_traj = {
        'N': 100, 'T': 10.0, 'dt': 0.01,
        'beta': 10.0, 'R_sp': 0.15, 'R_op': 0.15,
        'sigma_sp': 0.05, 'sigma_op': 0.05
    }
    
    for idx, alpha in enumerate(alpha_values_traj):
        current_params = base_params_traj.copy()
        current_params['alpha'] = alpha
        
        all_runs = []

        for _ in range(10):
            times, r_curve = run_trajectory(current_params, sample_rate=20)
            all_runs.append(r_curve)
            
        min_len = min(len(r) for r in all_runs)
        all_runs = np.array([r[:min_len] for r in all_runs])
        times = times[:min_len]
        
        mean_curve = np.mean(all_runs, axis=0)
        ax1.plot(times, mean_curve, label=f'α={alpha}', color=colors_traj[idx], lw=2)
        
    ax1.set_title("(a) H1 Phenomenon: Trajectories", fontsize=12)
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Assortativity (r)")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()



    print("Generating Part B: Convergence Time vs Alpha...")
    alpha_values_speed = np.array([10, 20, 30, 40, 50, 60, 80, 100])
    convergence_times = []
    errors = []
    
    base_params_speed = {
        'N': 100, 'dt': 0.01, 'beta': 10.0,
        'R_sp': 0.15, 'R_op': 0.15,
        'sigma_sp': 0.05, 'sigma_op': 0.05
    }
    
    for alpha in tqdm.tqdm(alpha_values_speed, leave=False):
        current_params = base_params_speed.copy()
        current_params['alpha'] = alpha
        
        times = []
        for _ in range(10):
            t = get_convergence_time(current_params, target_r=0.8)
            times.append(t)
        
        convergence_times.append(np.mean(times))
        errors.append(np.std(times))
        
    ax2.errorbar(alpha_values_speed, convergence_times, yerr=errors, fmt='-o', 
                 color='darkgreen', capsize=3, lw=2)
    ax2.set_title("(b) H1 Scaling: Speed vs Strength", fontsize=12)
    ax2.set_xlabel(r"Opinion Strength ($\alpha$)", fontsize=11)
    ax2.set_ylabel("Convergence Time ($T_{conv}$)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()




# testing h3
def test_h3():
    print("\n" + "="*60)
    print(">>> testing h3")
    print("="*60)
    
    critical_R_op = 0.156
    n_simulations = 100
    
    base_params = {
        'N': 100, 'dt': 0.01,
        'alpha': 40.0, 'beta': 10.0,
        'R_sp': 0.15, 'R_op': critical_R_op,
        'sigma_sp': 0.05, 'sigma_op': 0.05
    }
    
    print(f"Collecting data ({n_simulations} runs)...")
    results = []
    for _ in tqdm.tqdm(range(n_simulations)):
        results.append(get_assortativity(base_params))
    
    results = np.array(results)
    
 
    s = skew(results)
    k = kurtosis(results, fisher=True) + 3 
    
    BC = (s**2 + 1) / k
    
    print("-" * 30)
    print(f"Statistical Metrics:")
    print(f"Skewness: {s:.4f}")
    print(f"Kurtosis: {k:.4f}")
    print(f"Sarle's Bimodality Coefficient: {BC:.4f}")
    print("-" * 30)

    is_bimodal = BC > (5/9)
    judgment = "Bimodal" if is_bimodal else "Unimodal"
    
    print(f"Threshold: 0.555")
    print(f"Conclusion: Distribution is {judgment}")
    
    
    plt.figure(figsize=(8, 6))
    plt.hist(results, bins=20, range=(-0.1, 1.1), color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"H3 Test: Distribution at Critical Point\n$R_op={critical_R_op:.3f}$,$BC={BC:.3f}$ ({judgment})")
    plt.xlabel("Assortativity (r)")
    plt.ylabel("Count")
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.3)
    

    text_str = f"Sarle's BC = {BC:.3f}\n(>0.555 implies Bimodal)"
    plt.text(0.5, 5, text_str, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()


if __name__ == "__main__":
    #fig_5a()
    test_h1()
    #test_h3()