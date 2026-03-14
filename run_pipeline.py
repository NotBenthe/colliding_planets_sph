import os
import numpy as np
import pickle
import grav_planets
import save_collision
import plotter

def run_automated_collision(mode="head-on"):
    # 1. Setup Paths
    relaxed_planet = "./data/Planet300.dat" 
    collision_input = "./data/collision_setup.dat"
    output_pickle = f"results_{mode}.pkl"
    
    if not os.path.exists(relaxed_planet):
        print(f"Error: {relaxed_planet} not found.")
        return

    # 2. Configure Collision Geometry
    print(f"---> Preparing {mode} collision setup...")
    save_collision.save_collision(
        relaxed_data=relaxed_planet,
        output=collision_input,
        separation_r=5e9,
        impact_speed=13000,
        collision_axis=0
    )

    # 3. Run Simulation
    print(f"---> Running SPH Simulation for {mode}...")
    omega_vector = np.array([0.0, 0.0, 0.0])

    results = grav_planets.run_sph(
        h=0.015e9, 
        dim=3, 
        gamma=1.4, 
        t_final=200000, 
        steps=50, 
        filename=collision_input,
        omega=omega_vector 
    )

    # 4. Save results to pickle
    with open(output_pickle, "wb") as f:
        pickle.dump(results, f)
    print(f"---> Results saved to {output_pickle}")

    # 5. Generate Visualization
    print(f"---> Plotting {mode} snapshots...")
    t_snaps = np.linspace(results['t'].min(), results['t'].max(), 9)
    plotter.plot_maker_multi(
        results, 
        t_plots=t_snaps, 
        outfile=f"./figures/snapshots_{mode}.png"
    )
    
    print(f"Pipeline complete for {mode}. Check 'snapshots_{mode}.png'.")

if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.makedirs("./data")
    run_automated_collision(mode="head-on")