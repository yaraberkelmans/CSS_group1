import numpy as np
import matplotlib.pyplot as plt
import tqdm
from src.model import AgentBasedModel
from scipy.stats import skew, kurtosis

# this file is for testing h1 and h2

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

# =========================================================================================
# 1. test H1: phase transition
# =========================================================================================

# h1a: assortativity vs R_op (scatter plot)
def test_h1a_scatter():    
    R_op_values = np.linspace(0.05, 0.25, 50)
    n_simulations = 20 
    
    base_params = {
        'N': 100, 'T': 10.0, 'dt': 0.01,
        'alpha': 40.0, 'beta': 10.0,
        'R_sp': 0.15, 
        'sigma_sp': 0.05, 'sigma_op': 0.05
    }
    
    x_coords = [] # R_op
    y_coords = [] # Assortativity
    
    for r_op in tqdm.tqdm(R_op_values):
        current_params = base_params.copy()
        current_params['R_op'] = r_op
        
        for i in range(n_simulations):
            seed = np.random.randint(0, 100000)
            current_params['seed'] = seed
            
            model = AgentBasedModel(**current_params)
            model.run()
            
            r = model.global_assortativity()
            x_coords.append(r_op)
            y_coords.append(r)
            
    # plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords, y_coords, s=20, alpha=0.2, c='black', marker='o')
    
    plt.title(f"Discontinuous Phase Transition in Global Assortativity vs $R_{{op}}$\n(N={base_params['N']}, α={base_params['alpha']}, β={base_params['beta']}, σ={base_params['sigma_op']})")
    plt.xlabel("Opinion Interaction Radius ($R_{op}$)", fontsize=12)
    plt.ylabel("Global Assortativity ($r$)", fontsize=12)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.show()

# h1a: assortativity vs R_op (histogram)
def test_h1a_histogram():
    R_op_list = [0.15, 0.16, 0.17, 0.18]
    n_simulations = 50
    
    base_params = {
        'N': 100, 'dt': 0.01,
        'alpha': 40.0, 'beta': 10.0,
        'R_sp': 0.15, 
        'sigma_sp': 0.05, 'sigma_op': 0.05
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, r_op in enumerate(R_op_list):  
        results = []
        # run n simulations
        for _ in tqdm.tqdm(range(n_simulations), leave=False):
            seed = np.random.randint(0, 100000)
            
            current_params = base_params.copy()
            current_params['R_op'] = r_op
            current_params['seed'] = seed
            
            model = AgentBasedModel(T=10.0, **current_params)
            model.run()
            results.append(model.global_assortativity())
            
        ax = axes[idx]
        ax.hist(results, bins=15, range=(-0.1, 1.1), 
                color='skyblue', edgecolor='black', alpha=0.7)
        
        ax.set_title(f"$R_{{op}} = {r_op}$", fontsize=14, fontweight='bold')
        
        # calculate bimodality coefficient
        from scipy.stats import skew, kurtosis
        s = skew(results)
        k = kurtosis(results, fisher=True) + 3
        bc = (s**2 + 1) / k
        ax.text(0.05, 0.9, f"BC={bc:.2f}\n(>0.55 implies bimodal)", transform=ax.transAxes, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7))

    fig.suptitle(f"Evolution of Assortativity Distributions Across the Critical Transition Regime\n(N={base_params['N']}, α={base_params['alpha']}, β={base_params['beta']}, σ={base_params['sigma_op']})", fontsize=16)
    fig.text(0.5, 0.04, 'Global Assortativity (r)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()

# h1b: opinions vs R_op
def test_h1b():
    R_op_values = np.linspace(0.05, 0.25, 50)
    n_simulations = 10
    
    base_params = {
        'N': 100,
        'T': 10.0,
        'dt': 0.01,
        'alpha': 40.0,
        'beta': 10.0,
        'R_sp': 0.15,
        'sigma_sp': 0.05,
        'sigma_op': 0.05
    }
    
    # save plot points
    x_coords = [] # R_op
    y_coords = [] # theta
    
    for r_op in tqdm.tqdm(R_op_values):
        current_params = base_params.copy()
        current_params['R_op'] = r_op
        
        for i in range(n_simulations):
            seed = np.random.randint(0, 100000)
            current_params['seed'] = seed
            
            model = AgentBasedModel(**current_params)
            model.run()
            
            # record all agents opinions at steady state
            current_opinions = [agent.theta[0] for agent in model.agents]
        
            x_coords.extend([r_op] * len(current_opinions))
            y_coords.extend(current_opinions)
            
    # plotting
    plt.figure(figsize=(12, 7))
    
    plt.scatter(x_coords, y_coords, s=20, alpha=0.1, c='black', marker='o')
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    
    plt.title(f"Steady States distribution vs $R_{{op}}$\n(N={base_params['N']}, α={base_params['alpha']}, β={base_params['beta']}, σ={base_params['sigma_op']})", fontsize=14)
    plt.xlabel("Opinion Interaction Radius ($R_{op}$)", fontsize=12)
    plt.ylabel("Individual Agent Opinions ($\\theta$)", fontsize=12)
    
    plt.ylim(-1.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =========================================================================================
# 2. test H2: speed of emergence varying alpha and beta
# =========================================================================================
def test_h2a():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    param_str = (
        r"$N=100, \beta=10.0, R_{sp}=0.15, R_{op}=0.15, \sigma_{sp}=\sigma_{op}=0.05$"
    )
  
    # plot A: Assortativity vs Time
    alpha_values_traj = [5.0, 40.0, 80.0]
    colors_traj = ['#1f77b4', '#ff7f0e', '#d62728']
    
    base_params_traj = {
        'N': 100, 'T': 10.0, 'dt': 0.01,
        'beta': 10.0, 'R_sp': 0.15, 'R_op': 0.15,
        'sigma_sp': 0.05, 'sigma_op': 0.05
    }
    
    for idx, alpha in enumerate(alpha_values_traj):
        current_params = base_params_traj.copy()
        current_params['alpha'] = alpha
        
        all_runs = []
        for _ in range(20): # run 20 times for average
            times, r_curve = run_trajectory(current_params, sample_rate=20)
            all_runs.append(r_curve)
            
        min_len = min(len(r) for r in all_runs)
        all_runs = np.array([r[:min_len] for r in all_runs])
        times = times[:min_len]
        
        mean_curve = np.mean(all_runs, axis=0)
        std_curve = np.std(all_runs, axis=0)
        
        ax1.plot(times, mean_curve, label=r'$\alpha$=' + f'{alpha}', color=colors_traj[idx], lw=2.5)
        ax1.fill_between(times, mean_curve - std_curve, mean_curve + std_curve, color=colors_traj[idx], alpha=0.15)
    
    ax1.set_title(f"(a) Global Assortativity vs. Time (Varying $\\alpha$)\n{param_str}", fontsize=11, pad=10)
    ax1.set_xlabel("Time ($t$)", fontsize=12)
    ax1.set_ylabel("Global Assortativity ($r$)", fontsize=12)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=11, loc='lower right', title="Opinion Strength")

    # plot B: Convergence Time vs Opinion Strength alpha
    alpha_values_speed = np.linspace(1.0, 100, 10) 
    convergence_times = []
    errors = []
    
    base_params_speed = {
        'N': 100, 'dt': 0.01, 'beta': 10.0,
        'R_sp': 0.15, 'R_op': 0.15,
        'sigma_sp': 0.05, 'sigma_op': 0.05
    }
    
    target_r = 0.8
    
    for alpha in tqdm.tqdm(alpha_values_speed, leave=False):
        current_params = base_params_speed.copy()
        current_params['alpha'] = alpha
        
        times = []
        for _ in range(20):
            t = get_convergence_time(current_params, target_r=target_r)
            if t is not None:
                times.append(t)
            else:
                times.append(10.0)
        
        convergence_times.append(np.mean(times))
        errors.append(np.std(times))
        
    ax2.errorbar(alpha_values_speed, convergence_times, yerr=errors, fmt='-o', 
                 color='darkgreen', ecolor='darkgreen', capsize=4, lw=2, markersize=6)

    ax2.set_title(f"(b) Convergence Time vs. Opinion Strength\n{param_str}, Target $r={target_r}$", fontsize=11, pad=10)
    ax2.set_xlabel(r"Opinion Strength ($\alpha$)", fontsize=12)
    ax2.set_ylabel("Convergence Time ($T_{conv}$)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


def test_h2b():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    param_str = r"$N=100, \alpha=40.0, R_{sp}=0.15, R_{op}=0.15, \sigma=0.05$"

    # plot A: Assortativity vs Time
    beta_values_traj = [5, 10, 50]
    colors_traj = ['#1f77b4', '#ff7f0e', '#d62728']
    
    base_params = {
        'N': 100, 'T': 10.0, 'dt': 0.01,
        'alpha': 40.0,
        'R_sp': 0.15, 'R_op': 0.15,
        'sigma_sp': 0.05, 'sigma_op': 0.05
    }
    
    for idx, beta in enumerate(beta_values_traj):
        current_params = base_params.copy()
        current_params['beta'] = beta
        
        all_runs = []
        for _ in range(20):
            times, r_curve = run_trajectory(current_params, sample_rate=20)
            all_runs.append(r_curve)
            
        min_len = min(len(r) for r in all_runs)
        all_runs = np.array([r[:min_len] for r in all_runs])
        times = times[:min_len]
        
        mean_curve = np.mean(all_runs, axis=0)
        std_curve = np.std(all_runs, axis=0)
        
        ax1.plot(times, mean_curve, label=r'$\beta$=' + f'{beta}', color=colors_traj[idx], lw=2.5)
        ax1.fill_between(times, mean_curve - std_curve, mean_curve + std_curve, color=colors_traj[idx], alpha=0.15)
    
    ax1.set_title(f"(a) Global Assortativity vs. Time (Varying $\\beta$)\n{param_str}", fontsize=11, pad=10)
    ax1.set_xlabel("Time ($t$)", fontsize=12)
    ax1.set_ylabel("Global Assortativity ($r$)", fontsize=12)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=11, loc='lower right', title="Social Strength")


    # part B: Convergence Time vs Beta
    beta_values_speed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
    convergence_times = []
    errors = []
    
    target_r = 0.8 
    
    for beta in tqdm.tqdm(beta_values_speed, leave=False):
        current_params = base_params.copy()
        current_params['beta'] = beta
        
        times = []
        for _ in range(50):
            t = get_convergence_time(current_params, target_r=target_r)
            if t is not None:
                times.append(t)
            else:
                times.append(10.0) # manually set max time if not converged
        
        convergence_times.append(np.mean(times))
        errors.append(np.std(times))
        
    ax2.errorbar(beta_values_speed, convergence_times, yerr=errors, fmt='-o', 
                 color='purple', ecolor='purple', capsize=4, lw=2, markersize=6)
    
    ax2.set_title(f"(b) Convergence Time vs. Social Strength\n{param_str}, Target $r={target_r}$", fontsize=11, pad=10)
    ax2.set_xlabel(r"Social Strength ($\beta$)", fontsize=12)
    ax2.set_ylabel("Convergence Time ($T_{conv}$)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #test_h1a_histogram()
    #test_h1a_scatter()
    test_h2a()
    #test_h2b()
