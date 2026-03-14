import numpy as np
import pickle
with open("./data/data.pkl", "rb") as f:
        results = pickle.load(f)

print(results['t'][-1])
m = np.loadtxt("./data/Planet300_collision.dat", usecols=6)

x0 = results['x'][0]
v0 = results['v'][0]

planetA_idx = np.arange(301)
planetB_idx = np.arange(301, 602)

pos_i = x0[planetA_idx]
m_i   = m[planetA_idx]

com_i = np.average(pos_i, axis=0, weights=m_i)
pos_i -= com_i

M_i = np.sum(m_i)

def mass_radius(positions, masses, frac=0.99):
    r = np.linalg.norm(positions, axis=1)
    idx = np.argsort(r)
    r_sorted = r[idx]
    m_sorted = masses[idx]
    cum_mass = np.cumsum(m_sorted)
    return r_sorted[np.searchsorted(cum_mass, frac*cum_mass[-1])]

R_i = mass_radius(pos_i, m_i)

rho_i = M_i / ((4/3)*np.pi*R_i**3)

xf = results['x'][-1]
vf = results['v'][-1]
com_guess = np.average(xf, axis=0, weights=m)
xf_rel = xf - com_guess
vf_rel = vf - np.average(vf, axis=0, weights=m)
G = 6.67430e-11 

N = len(m)
phi = np.zeros(N)

for i in range(N):
    r_ij = np.linalg.norm(xf[i] - xf, axis=1)
    r_ij[i] = np.inf
    phi[i] = -G * np.sum(m / r_ij)
v2 = np.sum(vf_rel**2, axis=1)
E = 0.5 * m * v2 + m * phi
bound = E < 0
bound_idx = np.where(bound)[0]

pos_f = xf[bound_idx]
m_f   = m[bound_idx]

com_f = np.average(pos_f, axis=0, weights=m_f)
pos_f -= com_f

M_f = np.sum(m_f)
R_f = mass_radius(pos_f, m_f)
rho_f = M_f / ((4/3)*np.pi*R_f**3)

print(f' Initial Mass: {M_i:.3E}\n Initial Radius:{R_i:.3E}\n Initial Density: {rho_i:.3E}\n Final Mass: {M_f:.3E}\n Final Radius: {R_f:.3E}\n Final Density: {rho_f:.3E}\n')
print()