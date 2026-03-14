from solve_sph import SPHParameters, RK45Solver
import matplotlib.pyplot as plt
import sodshock

def get_hydro_states(sol, params, m):
    """Helper to reconstruct rho and p from the solver output."""
    N = m.size
    snapshots = []
    for i in range(len(sol.t)):
        x = sol.y[0:3*N, i].reshape((N, 3))
        v = sol.y[3*N:6*N, i].reshape((N, 3))
        e = sol.y[6*N:7*N, i]
        
        # Temporary instance to use density logic
        temp_phys = RK45Solver(params)
        rho = temp_phys.get_density(x)
        p = (params.gamma - 1.0) * rho * e
        snapshots.append({'x': x[:,0], 'v': v[:,0], 'e': e, 'rho': rho, 'p': p, 't': sol.t[i]})
    return snapshots

def plot_comparison(ax, state, params, var_name='rho'):
    # SPH data
    ax.scatter(state['x'], state[var_name], color='black', s=5, label='SPH')
    
    # Exact solution (Analytical)
    left, right = (1.0, 1.0, 0.0), (0.25, 0.1795, 0.0)
    _, _, values = sodshock.solve(left, right, (0, 1.0, 0.5), state['t'], params.gamma)
    
    x_ex = params.domain[0] + values['x'] * (params.domain[1] - params.domain[0])
    ax.plot(x_ex, values[var_name if var_name != 'v' else 'u'], 'r-', label='Exact')
    ax.set_title(f'Time: {state["t"]:.3f}')

# --- Main Execution ---
params = SPHParameters(steps=40)
solver = RK45Solver(params)
solution = solver.solve()
history = get_hydro_states(solution, params, solver.m)

# Plot intervals similar to your example
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
final_state = history[-1]

plot_comparison(axs[0,0], final_state, params, 'rho')
axs[0,0].set_ylabel('Density')

plot_comparison(axs[0,1], final_state, params, 'p')
axs[0,1].set_ylabel('Pressure')

plot_comparison(axs[1,0], final_state, params, 'v')
axs[1,0].set_ylabel('Velocity')

plot_comparison(axs[1,1], final_state, params, 'e')
axs[1,1].set_ylabel('Internal Energy')

plt.tight_layout()
plt.show()