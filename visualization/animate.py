import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import AgentBasedModel

def animate():    
    params = {
        'N': 100, 
        'T': 4,    # total length
        'dt': 0.01,   # time step size, could change the speed here
        'alpha': 40,   
        'beta': 3,    
        'R_sp': 0.15,    
        'R_op': 0.15,    
        'sigma_sp': 0.05,
        'sigma_op': 0.05,
        'seed': np.random.randint(0, 10000)
    }
    
    model = AgentBasedModel(**params)
    
    rng = np.random.default_rng(params['seed'])
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel("Social Space X")
    ax.set_ylabel("Social Space Y")
    ax.set_title(f"Co-evolution Dynamics\n(α={params['alpha']}, β={params['beta']}, R={params['R_sp']})")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    lines = LineCollection([], colors='gray', alpha=0.2, linewidths=0.6)
    ax.add_collection(lines)
    
    # initial scatter
    scat = ax.scatter([], [], c=[], cmap='coolwarm', s=60, edgecolors='black', vmin=-0.2, vmax=0.2)
    
    # color bar
    cbar = plt.colorbar(scat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Opinion (Theta)")
    
    # time tag
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.8))

    # update function for animation
    def update(frame):
        model.step(rng)
        current_time = frame * params['dt']
        
        XY = np.array([agent.x for agent in model.agents])
        Thetas = np.array([agent.theta[0] for agent in model.agents])
        
        # update scatter
        scat.set_offsets(XY)
        scat.set_array(Thetas)
        
        # update lines
        A_sp, _ = model.adjacency_matrix()
        
        # update line segments
        start_indices, end_indices = np.where(np.triu(A_sp, k=1))
        
        if len(start_indices) > 0:
            segments = []
            for start, end in zip(start_indices, end_indices):
                segments.append([XY[start], XY[end]])
            lines.set_segments(segments)
        else:
            lines.set_segments([])
        
        # update time text
        time_text.set_text(f"Time: {current_time:.2f}")
        
        return scat, lines, time_text

    n_frames = int(params['T'] / params['dt'])
    print(f"Generating {n_frames} frames...")
    
    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True, repeat=False)
    
    plt.tight_layout()
    
    plt.show()
    #ani.save('img/beta_30.gif', writer='pillow', fps=20)

if __name__ == "__main__":
    animate()