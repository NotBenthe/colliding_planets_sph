import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize

def plot_maker_multi(results, t_plots, outfile, ncols=3):
    t_arr = results['t']
    indices = [np.argmin(np.abs(t_arr - t)) for t in t_plots]
    
    e_all = np.concatenate([results['e'][i] for i in indices])
    norm = Normalize(vmin=e_all.min(), vmax=e_all.max())
    
    # --- DYNAMIC LIMITS ---
    # Find the max extent across all snapshots to keep the "camera" still
    all_x = np.concatenate([results['x'][idx][:, 0] for idx in indices])
    all_y = np.concatenate([results['x'][idx][:, 1] for idx in indices])
    max_range = max(np.abs(all_x).max(), np.abs(all_y).max()) * 1.1 
    
    nplots = len(indices)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    
    for ax, idx in zip(axes, indices):
        x_snap = results['x'][idx]
        e_snap = results['e'][idx]
        
        sc = ax.scatter(
             x_snap[:, 0],
             x_snap[:, 1],
            c=e_snap,
            cmap='inferno',
            norm=norm,
            s=8
        )
        
        ax.set_title(f't = {t_arr[idx]:.1f}s') # Simplified title for debugging
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        
        # Apply the dynamic limits here
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect('equal') # Keep planets circular

    for ax in axes[nplots:]:
        ax.remove()
    
    fig.colorbar(sc, ax=axes[:nplots], label='internal energy')
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Plot saved to {outfile}")
    
if __name__ == "__main__":
    with open("./data/data.pkl", "rb") as f:
        results = pickle.load(f)
    n_frames = 16
    plot_maker_multi(results, t_plots=np.linspace(648000, np.max(results['t']/16.5), n_frames), outfile='./figures/collision_offset.png', ncols=4)