import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.integrate import solve_ivp 
from matplotlib.colors import Normalize


# Make the axes better
plt.rcParams['xtick.minor.visible'], plt.rcParams['xtick.top'] = True,True 
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True 
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'

plt.style.use('seaborn-v0_8-colorblind')
plt.style.use('dark_background')

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

    #print(f"max: {np.max(Wret)}\nmin: {np.min(Wret)}")

    #plt.figure()
    #plt.imshow(Wret, cmap='inferno')
    #plt.show()

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
    #rho = np.sum(m * W, axis=1)
    rho = np.sum(m.flatten() * W, axis=1)
    return rho 

def grad_phi(dx, h):
    r_abs = np.linalg.norm(dx, axis=-1)
    dphi = np.zeros_like(r_abs)
    R = r_abs / h
    mask1 = (R >= 0.0) & (R < 1.0)
    mask2 = (R >= 1.0) & (R < 2.0)
    mask3 = (R >= 2.0)
    dphi[mask1] = 1/h**2*((4/3)*R[mask1]-(6/5)*R[mask1]**3+(1/2)*R[mask1]**4)
    dphi[mask2] = 1/h**2*((8/3)*R[mask2]-3*R[mask2]**2+(6/5)*R[mask2]**3-(1/6)*R[mask2]**4-1/(15*R[mask2]**2))
    #dphi[mask3] = 1/r_abs[mask3]**2
    r_safe = np.maximum(1e-20, r_abs)
    dphi[mask3] = 1/r_safe[mask3]**2
    return dphi

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

def sph_grav(x, v, l, e, m, h, dim, gamma):
    G = 6.6743e-11
    rho = compute_density(x, m, h, dim)

    if np.any(e < 0):
        print(f"**** negative energy {np.min(e)}")

    e_safe = np.maximum(1e-10, e)
    p = (gamma - 1.0) * rho * e_safe  # eq 8
    c = np.sqrt((gamma - 1.0) * e_safe)  # eq 9

    dx = x[:, None, :] - x[None, :, :]
    dW = grad_W_cubic_spline(dx, h, dim)

    Pi = artificial_viscosity(x, v, rho, c, h)

    t1 = p / rho**2 
    t2 = t1[:, None] + t1[None, :] + Pi 

    m_flat = m.flatten()

    dphi = grad_phi(dx, h)  # (301, 301)
    term1 = - np.sum(m_flat[None, : , None] * t2[:, :, None] * dW, axis=1)  # (1, 301, 3)    
    
    r_ij = np.linalg.norm(dx, axis=-1)
    r_ij_safe = np.where(r_ij == 0, 1.0, r_ij)
    np.fill_diagonal(r_ij_safe, 1)
    
    dxoverr = dx / r_ij_safe[:, :, None] 
    #dx_r = np.einsum('j, ij, ijk, ij->ik', m[:, 0], dphi, dx, 1/r_ij_safe)  # (301, 301)
    dx_r = np.einsum('j, ij, ijk->ik', m[:, 0], dphi, dxoverr)  # (301, 301)
    term2 = -G * dx_r
    #term2 = - G * np.sum(m_flat[None, :, None] * (dphi / r_ij_safe)[:, :, None] * dx, axis=1)

    dv = term1 + term2
    deltav = v[:, None, :] - v[None, :, :]
    #deltav = v[None, :, :] - v[:, None, :]

    de = 0.5 * np.einsum('j, ij, ijk, ijk->i', m_flat, t2, deltav, dW)
    #de = 0.5 * np.sum(m_flat[None, :] * t2 * np.sum(deltav * dW, axis=-1), axis=1)
    
    return dv, de, rho, p, l

def initial_conditions_grav(filename, gamma, omega):
    planet = np.loadtxt(filename)
    x, y, z, vx, vy, vz, m, rho, p = np.hsplit(planet, 9)
    xarr = np.hstack((x,y,z))
    com = np.average(xarr, axis=0, weights=m.ravel())
    xarr -= com
    varr = np.hstack((vx,vy,vz))
    larr = np.cross(omega, xarr)
    omega = np.asarray(omega).reshape(3,)
    spin = np.cross(xarr, omega)
    varr = varr + spin
    e = p/(rho*(gamma-1))
    #print(spin)
    #print("Max |v|:", np.max(np.linalg.norm(varr, axis=1)))
    #print("Max |r|:", np.max(np.linalg.norm(xarr, axis=1)))
    #print("Any NaNs in v?", np.isnan(varr).any())
    #print("Any NaNs in r?", np.isnan(xarr).any())
    return xarr, varr, larr, rho, p, e, m

def final_to_initial(filename, gamma, omega):
    final = np.loadtxt(filename, delimiter=",", dtype=float)

# concatenating x (N, 3), v (N, 3) and e(N, ) into 1d vector and back
def conc(x, v, l, e):
    N = x.shape[0]
    return np.concatenate([x.ravel(), v.ravel(), l.ravel(), e.ravel()])

def deconc(y, N):
    x = y[0:3*N].reshape((N, 3))
    v = y[3*N:6*N].reshape((N, 3))
    l = y[6*N:9*N].reshape((N, 3))
    e = y[9*N:10*N]
    return x, v, l, e 

def ode_grav(t, y, m, h, dim, gamma):
    # RHS for solve_ivp dy/dt = f(t, y)
    N = m.size 
    x, v, l, e = deconc(y, N)
    dv, de, _, _, l = sph_grav(x, v, l, e, m, h, dim, gamma)
    dxdt = v 
    dvdt = dv 
    dedt = de 
    return conc(dxdt, dvdt, l, dedt)

def run_sph(h, dim, gamma, t_final, steps, filename, omega): 
    # running SPH with RH45 
    # creating initial conditions
    x0, v0, l0, rho0, p0, e0, m = initial_conditions_grav(filename, gamma, omega)
    N = m.size
    t_span = (0, t_final)
    t_eval = np.linspace(0, t_final, steps+1)
    y0 = conc(x0, v0, l0, e0)
    sol = solve_ivp(ode_grav, t_span, y0, method='RK45', t_eval=t_eval, args=(m, h, dim, gamma)) 
                    #rtol=1e-4, atol=1e-6)
    print(sol.message)

    # extracting results
    t_num = sol.y.shape[1]
    x_all = sol.y[0:3*N, :].T.reshape(t_num, N, 3)
    v_all = sol.y[3*N:6*N, :].T.reshape(t_num, N, 3)
    l_all = sol.y[6*N:9*N, :].T.reshape(t_num, N, 3)
    e_all = sol.y[9*N:10*N, :].T
    
    res = {
        't': sol.t,
        'x': x_all,
        'v': v_all,
        'e': e_all,
    } 

    print("finished running SPH")
    with open("./data/data.pkl", "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

    return res 

def movie_maker(results, filename, fps=180):
    x_all = results['x']
    t_arr = results['t']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(x_all[0, :, 0], x_all[0, :, 1], x_all[0, :, 2], 
                      c='white', s=2, alpha=0.6)
    limit = np.max(np.abs(x_all))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def animate(i):
        ax.clear()
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        
        scat = ax.scatter(x_all[i, :, 0], x_all[i, :, 1], x_all[i, :, 2], 
                          c='white', s=2, alpha=0.6)
        ax.set_title(f"Time: {t_arr[i]:.3f}")
        return scat,

    anim = animation.FuncAnimation(fig, animate, frames=len(t_arr/10), interval=1000/fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close(fig)
    print(f"saved movie as {filename}")

def plot_maker(results, t_plot, filename):
    t_arr = results['t']
    # plotting at index closest to t_plot
    idx = np.argmin(np.abs(t_arr - t_plot))
    x_snap = results['x'][idx] 
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    p = ax.scatter(x_snap[:, 0], x_snap[:, 1], 
                   c=results['e'][idx], cmap='inferno', s=8)
    
    fig.colorbar(p, label='internal energy')
    ax.set_title(f'snapshot at t = {t_arr[idx]:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"saved image as {filename}")

def plot_maker_multi(results, t_plots, filename, ncols=3):
    t_arr = results['t']
    
    indices = [np.argmin(np.abs(t_arr - t)) for t in t_plots]
    
    e_all = np.concatenate([results['e'][i] for i in indices])
    norm = Normalize(vmin=e_all.min(), vmax=e_all.max())
    
    nplots = len(indices)
    nrows = int(np.ceil(nplots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4*ncols, 4*nrows),
                             constrained_layout=True)
    
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
        
        ax.set_title(f't = {t_arr[idx]:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-5e9, 5e9)
        ax.set_ylim(-5e9, 5e9)

    for ax in axes[nplots:]:
        ax.remove()
    
    fig.colorbar(sc, ax=axes[:nplots], label='internal energy')
    plt.suptitle(r'Rotationless Collision, 13 $\text{km}\cdot\text{s}^{-1}$', fontsize=20)
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"saved image as {filename}")


def run_everything(h, dim, gamma, t_final, steps, domain, omega, filename=None):
     # running SPH
    results = run_sph(h, dim, gamma, t_final, steps, filename, omega)
    n, n_p, _ = results['x'].shape
    df = pd.DataFrame({
        'time': np.repeat(results['t'], n_p),
        'particle_id': np.tile(np.arange(n_p), n),
        'x': results['x'][:, :, 0].ravel(),
        'y': results['x'][:, :, 1].ravel(),
        'z': results['x'][:, :, 2].ravel(),
        'vx': results['v'][:, :, 0].ravel(),
        'vy': results['v'][:, :, 1].ravel(),
        'vz': results['v'][:, :, 2].ravel(),
        'energy': results['e'].ravel()
    })
    #df.to_csv("results.csv", index=False)
    n_frames = 9
    plot_maker_multi(results, t_plots=np.linspace(np.min(results['t']), np.max(results['t']/100), n_frames), filename="./figures/snapshots.png")
    #movie_maker(results, "one_planet.mp4")


if __name__ == "__main__":
    domain = [-0.6, 0.6]
    h = 3.9e7
    dim = 3
    gamma = 1.4 
    dt = 150
    steps = 100000
    t_final = dt * steps
    filename = './data/Planet300_collision.dat'

    # rotation does not work yet
    period = 20 * 60 * 60 # seconds
    omega_z = 0 #2 * np.pi / period 
    omega = [0, 0, omega_z]

    run_everything(h, dim, gamma, t_final, steps, domain, omega, filename)