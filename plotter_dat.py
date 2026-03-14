import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_dat_file(filename, outfile):
    # Load the 9-column data
    # Columns: x, y, z, vx, vy, vz, m, rho, p
    data = np.loadtxt(filename)
    
    x = data[:, 0]
    y = data[:, 1]
    p = data[:, 8]  # Pressure
    rho = data[:, 7] # Density
    
    # Calculate internal energy (e = p / (rho * (gamma - 1)))
    # Assuming gamma = 1.4 for now
    gamma = 1.4
    internal_energy = p / (rho * (gamma - 1))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Normalize color based on internal energy
    norm = Normalize(vmin=internal_energy.min(), vmax=internal_energy.max())
    
    sc = ax.scatter(
        x, y, 
        c=internal_energy, 
        cmap='inferno', 
        norm=norm, 
        s=5, 
        alpha=0.8
    )
    
    # Set limits based on the separation we used (approx 5e9)
    limit = 6e9 
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    fig.colorbar(sc, label='Internal Energy')
    plt.title(f'Initial Condition: {filename}')
    
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {outfile}")

if __name__ == "__main__":
    # Change the filename to match your generated .dat file
    plot_dat_file('./data/Planet300_collision.dat', './data/planet300_collision.png')