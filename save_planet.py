import numpy as np
import pandas as pd

'''
Program that takes the final state from results.csv and saves it as a new, relaxed initial state
'''

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

    #print(f'max: {np.max(Wret)}\nmin: {np.min(Wret)}')

    #plt.figure()
    #plt.imshow(Wret, cmap='inferno')
    #plt.show()

    return Wret 

def compute_density (x, m, h, dim):
    # \rho_i = \sum_j m_j W_{ij}
    dx = x[:, None, :] - x[None, :, :]
    r = np.linalg.norm(dx, axis=-1)
    W = W_cubic_spline(r, h, dim)
    #rho = np.sum(m * W, axis=1)
    rho = np.sum(m.flatten() * W, axis=1)
    return rho 

def save(csv, initial, output, h, dim=3, gamma=1.4):

    planet = np.loadtxt(initial)
    x0, y0, z0, vx0, vy0, vz0, m0, rho0, p0 = np.hsplit(planet, 9)

    m0 = m0.flatten()

    n_particles_initial = len(m0)

    df = pd.read_csv(csv)

    final_time = df['time'].max()
    df_final = df[df['time'] == final_time].copy()

    df_final = df_final.sort_values('particle_id')

    n_particles_final = len(df_final)

    if n_particles_final != n_particles_initial:
        raise ValueError(
            f'Particle count mismatch: '
            f'{n_particles_final} (final step) vs '
            f'{n_particles_initial} (initial)'
        )

    x = df_final[['x', 'y', 'z']].values
    v = df_final[['vx', 'vy', 'vz']].values
    e = df_final['energy'].values

    rho = compute_density(x, m0, h, dim)

    p = (gamma - 1.0) * rho * e

    output_array = np.column_stack([x, v, m0, rho, p])

    np.savetxt(output, output_array)

    print(f'Saved {n_particles_final} particles to {output}')

if __name__ == '__main__':
    save(csv='./results/results.csv', initial='./data/Planet300.dat', output='./data/Planet300_relaxed.dat', h=3.9e7, dim=3, gamma=1.4)