import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.integrate import solve_ivp 
import sodshock

# Make the axes better
plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.top'] = True,True 
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True 
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'

# Make the font look nicer
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{txfonts}"
plt.rcParams['font.size'] = 18 

plt.style.use('seaborn-v0_8-colorblind')
#plt.style.use('dark_background')

def W_cubic_spline (r, h, dim):
    W = np.zeros_like(r)
    R = r / h
    if dim == 1:
        alpha = 1/h
    elif dim == 3:
        alpha = 3/(2 * np.pi * h**3)

    mask1 = (R >= 0.0) & (R < 1.0)
    mask2 = (R >= 1.0) & (R < 2.0)

    W[mask1] = 2/3 - R[mask1]**2 + 1/2 * R[mask1]**3
    W[mask2] = 1/6 * (2-R[mask2])**3

    Wret = alpha * W
    return Wret 

def grad_W_cubic_spline(dx, h, dim):
    dW = np.zeros_like(dx)
    if dim == 1:
        alpha = 1/h
    elif dim == 3:
        alpha = 3/(2 * np.pi * h**3)

    r = np.linalg.norm(dx, axis=-1)
    R = r / h

    r_safe = np.where(r==0, 1.0, r)

    F = np.zeros_like(R)

    mask1 = (R >= 0.0) & (R < 1.0)
    mask2 = (R >= 1.0) & (R < 2)

    F[mask1] = alpha * (-2 + 3/2 * R[mask1]) / h**2 
    F[mask2] = -alpha * 1/2 * (2 - R[mask2])**2 /(h * r_safe[mask2])
    
    dW = F[:, :, None] * dx

    return dW 

def compute_density (x, m, h, dim):
    # \rho_i = \sum_j m_j W_{ij}
    dx = x[:, None, :] - x[None, :, :]
    r = np.linalg.norm(dx, axis=-1)
    W = W_cubic_spline(r, h, dim)
    rho = np.sum(m[None, :] * W, axis=1)
    return rho 

def artificial_viscosity(x, v, rho, c, h):
    dx = x[:, None, :] - x[None, :, :]
    dv = v[:, None, :] - v[None, :, :]

    #r2 = np.sum(dx, axis=-1)
    r2 = np.sum(dx**2, axis=-1)
    vdx = np.sum(dv*dx, axis=-1) 
    Pi = np.zeros_like(r2)

    mask = vdx < 0.0
    alpha_pi = 1.0
    beta_pi = 1.0
    psi = 0.1 * h 
    #phi_ij = h * vdx / (r2**2 + psi**2)
    phi_ij = h * vdx / (r2 + psi**2)

    rho_ij = 1/2 * (rho[:, None] + rho[None, :])
    c_ij = 1/2 * (c[:, None] + c[None, :]) 

    Pi[mask] = (-alpha_pi * c_ij[mask] * phi_ij[mask] + beta_pi * phi_ij[mask]**2)/(rho_ij[mask])

    return Pi 

def sph(x, v, e, m, h, dim, gamma):
    # computing dv/dt, de/dt using SPH
    N = x.shape[0]
    rho = compute_density(x, m, h, dim)
    p = (gamma - 1.0) * rho * e  # eq 8
    c = np.sqrt((gamma - 1.0) * e)  # eq 9

    dx = x[:, None, :] - x[None, :, :]
    dW = grad_W_cubic_spline(dx, h, dim)

    Pi = artificial_viscosity(x, v, rho, c, h)

    t1 = p / rho**2 
    t2 = t1[:, None] + t1[None, :] + Pi 

    dv = - np.sum(m[None, : , None] * t2[..., None] * dW, axis=1)

    deltav = v[:, None, :] - v[None, :, :]
    vdW = np.sum(deltav * dW, axis=-1)
    de = 0.5 * np.sum(m[None, :] * t2 * vdW, axis=1)
    return dv, de, rho, p

def initial_conditions_sod():
    N_l = 320
    N_r = 80
    dx_l = 0.001875
    dx_r = 0.0075
    x_l = np.linspace(-0.6 + dx_l / 2, 0.0 - dx_l/2, N_l)
    x_r = np.linspace(0.0 + dx_r / 2, 0.6 - dx_r/2, N_r)

    xs = np.concatenate([x_l, x_r])
    N = len(xs) 

    x = np.zeros((N, 3))
    x[:, 0] = xs 
    v = np.zeros((N, 3))

    m = np.full(N, 0.001875)
    rho = np.concatenate([np.full(N_l, 1.0), np.full(N_r, 0.25)])
    p = np.concatenate([np.full(N_l, 1.0), np.full(N_r, 0.1795)])
    e = np.concatenate([np.full(N_l, 2.5), np.full(N_r, 1.795)])

    return x, v, rho, p, e, m

# concatenating x (N, 3), v (N, 3) and e(N, ) into 1d vector and back
def conc(x, v, e):
    return np.concatenate([x.ravel(), v.ravel(), e.ravel()])

def deconc(y, N):
    x = y[0:3*N].reshape((N, 3))
    v = y[3*N:6*N].reshape((N, 3))
    e = y[6*N:7*N]
    return x, v, e 

def ode(t, y, m, h, dim, gamma):
    # RHS for solve_ivp dy/dt = f(t, y)
    N = m.size 
    x, v, e = deconc(y, N)
    dv, de, _, _ = sph(x, v, e, m, h, dim, gamma)
    dxdt = v 
    dvdt = dv 
    dedt = de 
    return conc(dxdt, dvdt, dedt)

def run_sph(h, dim, gamma, t_final, steps): 
    # running SPH with RH45 
    # creating initial conditions
    x0, v0, rho0, p0, e0, m = initial_conditions_sod()
    N = m.size
    t_span = (0, t_final)
    t_eval = np.linspace(0, t_final, steps+1)
    y0 = conc(x0, v0, e0)

    # running solve_ivp
    sol = solve_ivp(ode, t_span, y0, method='RK45', t_eval=t_eval, args=(m, h, dim, gamma), rtol=1e-8, atol=1e-8)
    print("solved the ivp")

    # extract results
    t_num = sol.y.shape[1]
    x_all = sol.y[0:3*N, :].T.reshape(t_num, N, 3)
    v_all = sol.y[3*N:6*N, :].T.reshape(t_num, N, 3)
    e_all = sol.y[6*N:7*N, :].T
    
    res = {
        't': sol.t,
        'x': x_all[:, :, 0],
        'v': v_all[:, :, 0],
        'e': e_all,
    } 

    return res 

def movie_maker(snapshots, t_arr, filename, domain, fps=25):
    fig, ax = plt.subplots(figsize=(7, 5))
    line, = ax.plot([], [], 'b.', markersize=3)
    ax.set_xlim(domain)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\rho$")

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x, _, _, rho, _ = snapshots[i]
        x1d = x[:, 0]
        order = np.argsort(x1d)
        line.set_data(x1d[order], rho[order])
        ax.set_title(f"SPH Sod Shock Tube, $t = {t_arr[i]:.3f}$")
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(snapshots), interval=50, blit=True)

    writer = animation.FFMpegWriter(fps=fps)
    # Using bbox_inches='tight' prevents the video axes from being cut off
    anim.save(filename, writer=writer) 
    plt.close(fig)

def plot_density_evolution(snapshots, t_arr, domain):
    indices = [0, len(t_arr)//4, len(t_arr)//2, len(t_arr)-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        x, _, _, rho, _ = snapshots[idx]
        x1d = x[:, 0]
        order = np.argsort(x1d)
        
        axes[i].plot(x1d[order], rho[order], 'b.', markersize=4)
        axes[i].set_title(f"Time $t = {t_arr[idx]:.3f}$")
        axes[i].set_xlim(domain)
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].set_xlabel(r"$x$")
        if i % 2 == 0: axes[i].set_ylabel(r"Density $\rho$")

    plt.tight_layout()
    plt.savefig("./figures/density_evolution.png", dpi=200)
    print("Saved density_evolution.png")
    plt.close()

def exact_sod(x_exact, t, domain, gamma):
    x_min, x_max = domain
    length = x_max - x_min 

    # p, rho, v
    left = (1.0, 1.0, 0.0)
    right = (0.1795, 0.25, 0.0)
    geo = (0, 1, 0.5) 

    # get exact sod values
    _, _, values = sodshock.solve(left_state=left, right_state=right, geometry=geo, t=t, gamma=gamma)

    x_val = values['x']
    p_val = values['p']
    rho_val = values['rho']
    u_val = values['u']

    x_vals = x_min + x_val * length 
    p_ex = np.interp(x_exact, x_vals, p_val)
    rho_ex = np.interp(x_exact, x_vals, rho_val)
    u_ex = np.interp(x_exact, x_vals, u_val)
    e_ex = p_ex / ((gamma - 1.0) * rho_ex) 
    return rho_ex, p_ex, e_ex, u_ex 

def plot_maker(snapshots, t_plot, t_arr, domain, gamma):
    # Find closest snapshot in time
    idx = np.argmin(np.abs(t_arr - t_plot))
    x, v, e, rho, p = snapshots[idx]
    x1d = x[:, 0]
    order = np.argsort(x1d)
    x_sorted = x1d[order]
    rho_sorted = rho[order]
    p_sorted = p[order]
    e_sorted = e[order]
    v_sorted = v[order, 0]

    # Exact solution on a fine grid
    x_exact = np.linspace(domain[0], domain[1], 1000)
    rho_ex, p_ex, e_ex, u_ex = exact_sod(x_exact, t_arr[idx], domain, gamma)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_rho, ax_p = axes[0]
    ax_e, ax_v = axes[1]

    # Density
    ax_rho.plot(x_sorted, rho_sorted, 'b.', label="SPH")
    ax_rho.plot(x_exact, rho_ex, 'r-', label="Exact")
    ax_rho.set_ylabel("density")
    ax_rho.legend(loc="best")
    ax_rho.set_xlim(domain)
    ax_rho.set_xlabel("x")

    # Pressure
    ax_p.plot(x_sorted, p_sorted, 'b.', label="SPH")
    ax_p.plot(x_exact, p_ex, 'r-', label="Exact")
    ax_p.set_ylabel("pressure")
    ax_p.set_xlim(domain)
    ax_p.set_xlabel("x")

    # Internal energy
    ax_e.plot(x_sorted, e_sorted, 'b.', label="SPH")
    ax_e.plot(x_exact, e_ex, 'r-', label="Exact")
    ax_e.set_ylabel("internal energy")
    ax_e.set_xlabel("x")
    ax_e.set_xlim(domain)

    # Velocity
    ax_v.plot(x_sorted, v_sorted, 'b.', label="SPH")
    ax_v.plot(x_exact, u_ex, 'r-', label="Exact")
    ax_v.set_ylabel("velocity")
    ax_v.set_xlabel("x")
    ax_v.set_xlim(domain)

    #plt.tight_layout()
    plt.suptitle(f"Sod Shock Tube Problem t = {t_arr[idx]:.3f}")
    plt.savefig("./figures/sod_sph.png", dpi=200)
    plt.close()

def run_everything(h, dim, gamma, t_final, steps, domain):
    # running SPH
    results = run_sph(h, dim, gamma, t_final, steps)
    _, _, _, _, _, m = initial_conditions_sod() 
    
    # prepare positions 
    t_steps = results['x'].shape[0]
    Ns = results['x'].shape[1]
    x = np.zeros((t_steps, Ns, 3))
    x[:, :, 0] = results['x']

    dx = x[:, :, None, :] - x[:, None, :, :]
    r = np.linalg.norm(dx, axis=-1)

    W = W_cubic_spline(r, h, dim)
    rho_all = np.sum(m[None, None, :] * W, axis=2)
    p_all = (gamma - 1.0) * rho_all * results['e']

    snapshots = [(x[i], results['v'][i,:,None], results['e'][i], rho_all[i], p_all[i]) for i in range(t_steps)]
    t_arr = results['t']
    
    # running everything
    plot_density_evolution(snapshots, t_arr, domain)
    plot_maker(snapshots, t_final, t_arr, domain, gamma)
    movie_maker(snapshots, t_arr, "./figures/sod_sph.mp4", domain)

if __name__ == "__main__":
    # running the code
    domain = [-0.6, 0.6]
    h = 0.015
    dim = 1
    gamma = 1.4
    dt = 0.005
    steps = 40
    t_final = dt * steps
    run_everything(h, dim, gamma, t_final, steps, domain)
