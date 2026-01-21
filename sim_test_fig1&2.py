import numpy as np
import matplotlib.pyplot as plt
from model import AgentBasedModel
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# for fig 1 & 2

def plot_snapshot(model, title):

    final_x = np.array([a.x for a in model.agents])

    final_theta = np.array([a.theta[0] for a in model.agents])
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    sc = ax.scatter(final_x[:, 0], final_x[:, 1], c=final_theta, 
                    cmap='coolwarm', vmin=-1, vmax=1, 
                    edgecolor='k', s=60, alpha=0.8)
    
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Opinion (Theta)')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Social Space X")
    ax.set_ylabel("Social Space Y")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.grid(True, linestyle='--', alpha=0.3)


    ax_inset = inset_axes(ax, width="35%", height="25%", loc='upper right')
    
    ax_inset.hist(final_theta, bins=15, range=(-1, 1), 
                  color='gray', edgecolor='white', alpha=0.7)
    
    ax_inset.set_xlim(-1, 1)
    ax_inset.set_title("Opinion Dist.", fontsize=9)
    ax_inset.set_yticks([])
    ax_inset.tick_params(axis='x', labelsize=8)
    
    plt.tight_layout()
    plt.show()


def run_experiment(case_name):
    
    print(f"--- Running Experiment: {case_name} ---")
    
    if case_name == "Fig 1a":
        params = {'alpha': 40.0, 'beta': 10.0}
    elif case_name == "Fig 1b":
        params = {'alpha': 40.0, 'beta': 40.0}
    elif case_name == "Fig 2a":
        params = {'alpha': 10.0, 'beta': 50.0}
    elif case_name == "Fig 2b":
        params = {'alpha': 100.0, 'beta': 50.0}
    else:
        print("Unknown case")
        return

    model = AgentBasedModel(
        N=100, 
        T=2.5, 
        dt=0.01,
        R_sp=0.15, 
        R_op=0.15, 
        sigma_sp=0.05, 
        sigma_op=0.05,
        **params
    )

    seed = np.random.randint(10000) 
    model.run(seed=seed)
    
    plot_snapshot(model, title=f"{case_name} (α={params['alpha']}, β={params['beta']})")

if __name__ == "__main__":
    run_experiment("Fig 1a")
    run_experiment("Fig 1b")
    run_experiment("Fig 2a")
    run_experiment("Fig 2b")