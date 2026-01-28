import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
from model import AgentBasedModel

def run_trajectory(model, sample_rate=10):
    times = []
    assortativities = []
    steps = int(model.T / model.dt)
    rng = np.random.default_rng(model.seed) # seed from the model

    times.append(0)
    assortativities.append(model.global_assortativity())
    
    for step in range(1, steps + 1):
        model.step(rng)
        
        if step % sample_rate == 0:
            times.append(step * model.dt)
            assortativities.append(model.global_assortativity())
            
    return np.array(times), np.array(assortativities)


# experiment 1: stubborn agents
def stubborn_agent_experiment():
    print("\n" + "="*60)
    print(">>> Run Stubborn Agents Experiment")
    print("="*60)
    
    R_op_consensus = 0.30
    
    # proportions of stubborn agents to test
    fractions = np.linspace(0.0, 0.20, 10)
    
    final_rs_mean = []
    final_rs_std = []
    
    for frac in tqdm.tqdm(fractions, desc="Processing:"):
        rs_current_frac = []
        
        # run multiple trials for each fraction
        for i in range(10):
            seed = np.random.randint(0, 100000)
            params = {
                'N': 150, 'T': 15.0, 'dt': 0.01,
                'alpha': 40.0, 'beta': 10.0,
                'R_sp': 0.15, 'R_op': R_op_consensus,
                'sigma_sp': 0.05, 'sigma_op': 0.05,
                'seed': seed
            }
            model = AgentBasedModel(**params)
            
            # pick the stubborn agents
            n_stubborn = int(model.N * frac)
            stubborn_ids = np.random.choice(model.N, n_stubborn, replace=False)
            
            steps = int(model.T / model.dt)
            rng = np.random.default_rng(seed)
            
            for _ in range(steps):
                # run one step
                model.step(rng)
                
                # enforce stubborn agents' opinions
                if n_stubborn > 0:
                    half = n_stubborn // 2
                    
                    # make half stubborn agents' opinion = -1
                    for idx in stubborn_ids[:half]:
                        model.agents[idx].theta = np.full(model.m, -1.0)
                    
                    # the other half = +1
                    for idx in stubborn_ids[half:]:
                        model.agents[idx].theta = np.full(model.m, 1.0)
            
            rs_current_frac.append(model.global_assortativity())
            
        final_rs_mean.append(np.mean(rs_current_frac))
        final_rs_std.append(np.std(rs_current_frac))
        
    # plotting
    plt.figure(figsize=(8, 6))
    plt.errorbar(fractions * 100, final_rs_mean, yerr=final_rs_std, fmt='-o', color='firebrick')
    plt.title(f"Stubborn Agents Experiment (Baseline R_op={R_op_consensus})")
    plt.xlabel("Fraction of Stubborn Agents (%)")
    plt.ylabel("Final Assortativity")
    plt.grid(True)
    
    plt.show()

# experiment 2: influencers
def influencer_experiment():
    print("\n" + "="*60)
    print(">>> Run Influencer Experiment")
    print("="*60)
    
    R_op_polar = 0.15
    R_sp = 0.15
    
    params = {
        'N': 100, 'T': 10.0, 'dt': 0.01,
        'alpha': 40.0, 'beta': 10.0,
        'R_sp': R_sp, 'R_op': R_op_polar,
        'sigma_sp': 0.05, 'sigma_op': 0.05
    }
    
    n_runs = 10
    
    baseline_curves = []
    influencer_curves = []
    
    # a. baseline condition: random initial positions
    for i in range(n_runs):
        params['seed'] = i # fixed seed for comparison
        model = AgentBasedModel(**params)
        times, r_curve = run_trajectory(model)
        baseline_curves.append(r_curve)
        
    # b. influencer condition
    for i in range(n_runs):
        params['seed'] = i
        model = AgentBasedModel(**params)
        
        # determine number of influencers
        n_inf = int(model.N * 0.05)
        if n_inf < 2: n_inf = 2
        
        # choose influencer agents
        inf_ids = np.random.choice(model.N, n_inf, replace=False)
        half = n_inf // 2
        
        # modify their positions and opinions
        # first half influencers
        for idx in inf_ids[:half]:
            model.agents[idx].x = np.array([-R_sp/2, -R_sp/2]) # put them in bottom-left corner
            model.agents[idx].theta = np.full(model.m, -1.0)   # set opinion to -1
            
        # second half
        for idx in inf_ids[half:]:
            model.agents[idx].x = np.array([R_sp/2, R_sp/2])   # put them in top-right corner
            model.agents[idx].theta = np.full(model.m, 1.0)    # set opinion to +1
        
        times, r_curve = run_trajectory(model)
        influencer_curves.append(r_curve)
        
    # align lengths
    min_len = min([len(c) for c in baseline_curves] + [len(c) for c in influencer_curves])
    times = times[:min_len]
    baseline_curves = [c[:min_len] for c in baseline_curves]
    influencer_curves = [c[:min_len] for c in influencer_curves]

    mean_base = np.mean(baseline_curves, axis=0)
    mean_inf = np.mean(influencer_curves, axis=0)

    # plotting
    plt.figure(figsize=(10, 6))
    plt.plot(times, mean_base, label='Normal (Random Start)', color='blue', linestyle='--')
    plt.plot(times, mean_inf, label='With Influencers (Extreme Start)', color='red', linewidth=3)
    
    plt.title("Do Influencers Accelerate Polarization?")
    plt.xlabel("Time")
    plt.ylabel("Assortativity")
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    stubborn_agent_experiment()
    influencer_experiment()