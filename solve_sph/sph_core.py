import numpy as np
from scipy.integrate import solve_ivp

class SPHParameters:
    """Holds physical constants and grid parameters for the SPH simulation."""
    def __init__(self, h=0.015, dim=1, gamma=1.4, dt=0.005, steps=40, domain=[-0.6, 0.6]):
        self.h = h
        self.dim = dim
        self.gamma = gamma
        self.dt = dt
        self.T = dt * steps
        self.steps = steps
        self.domain = domain

    def get_initial_conditions(self):
        """Returns the standard Sod Shock Tube initial state."""
        N_l, N_r = 320, 80
        dx_l, dx_r = 0.001875, 0.0075
        
        x_l = np.linspace(self.domain[0] + dx_l/2, 0.0 - dx_l/2, N_l)
        x_r = np.linspace(0.0 + dx_r/2, self.domain[1] - dx_r/2, N_r)
        xs = np.concatenate([x_l, x_r])
        
        N = len(xs)
        x = np.zeros((N, 3)); x[:, 0] = xs
        v = np.zeros((N, 3))
        m = np.full(N, 0.001875)
        e = np.concatenate([np.full(N_l, 2.5), np.full(N_r, 1.795)])
        
        return x, v, e, m

class SPHSolver:
    """Base class for SPH physics calculations."""
    def __init__(self, parameters):
        self.p = parameters
        self.x, self.v, self.e, self.m = self.p.get_initial_conditions()
        self.N = self.m.size
        self.t = 0.0

    def kernel(self, r):
        """Cubic Spline Kernel."""
        W = np.zeros_like(r)
        R = r / self.p.h
        alpha = 1.0/self.p.h if self.p.dim == 1 else 3.0/(2.0*np.pi*self.p.h**3)
        mask1, mask2 = (R < 1.0), (R >= 1.0) & (R < 2.0)
        W[mask1] = 2/3 - R[mask1]**2 + 0.5*R[mask1]**3
        W[mask2] = 1/6 * (2 - R[mask2])**3
        return alpha * W

    def grad_kernel(self, r_vec):
        """Gradient of the Cubic Spline Kernel."""
        dW = np.zeros_like(r_vec)
        alpha = 1.0/self.p.h if self.p.dim == 1 else 3.0/(2.0*np.pi*self.p.h**3)
        r_abs = np.linalg.norm(r_vec, axis=-1)
        R = r_abs / self.p.h
        mask1, mask2 = (R < 1.0), (R >= 1.0) & (R < 2.0)
        
        t1 = alpha * (-2 + 1.5 * R[mask1]) / self.p.h**2
        dW[mask1] = t1[:, None] * r_vec[mask1]
        t2 = -alpha * 0.5 * (2 - R[mask2])**2 / (self.p.h * r_abs[mask2])
        dW[mask2] = t2[:, None] * r_vec[mask2]
        return dW

    def get_density(self, x):
        dx = x[:, None, :] - x[None, :, :]
        r = np.linalg.norm(dx, axis=-1)
        return np.sum(self.m[None, :] * self.kernel(r), axis=1)

    def get_viscosity(self, x, v, rho, c):
        dx = x[:, None, :] - x[None, :, :]
        dv = v[:, None, :] - v[None, :, :]
        vdx = np.sum(dv * dx, axis=-1)
        mask = vdx < 0
        
        phi = (self.p.h * vdx) / (np.sum(dx**2, axis=-1) + (0.1 * self.p.h)**2)
        rho_avg = 0.5 * (rho[:, None] + rho[None, :])
        c_avg = 0.5 * (c[:, None] + c[None, :])
        
        Pi = np.zeros((self.N, self.N))
        Pi[mask] = (-1.0 * c_avg[mask] * phi[mask] + 1.0 * phi[mask]**2) / rho_avg[mask]
        return Pi

    def rhs(self, t, y):
        # Unpack state
        x = y[0:3*self.N].reshape((self.N, 3))
        v = y[3*self.N:6*self.N].reshape((self.N, 3))
        e = y[6*self.N:7*self.N]

        rho = self.get_density(x)
        p = (self.p.gamma - 1.0) * rho * e
        c = np.sqrt((self.p.gamma - 1.0) * e)

        dx_mat = x[:, None, :] - x[None, :, :]
        dW = self.grad_kernel(dx_mat)
        Pi = self.get_viscosity(x, v, rho, c)

        t_term = (p / rho**2)[:, None] + (p / rho**2)[None, :] + Pi
        dv = -np.sum(self.m[None, :, None] * t_term[..., None] * dW, axis=1)
        
        vdW = np.sum((v[:, None, :] - v[None, :, :]) * dW, axis=-1)
        de = 0.5 * np.sum(self.m[None, :] * t_term * vdW, axis=1)

        return np.concatenate([v.ravel(), dv.ravel(), de.ravel()])

class RK45Solver(SPHSolver):
    """Specific implementation using Scipy's RK45."""
    def solve(self):
        y0 = np.concatenate([self.x.ravel(), self.v.ravel(), self.e.ravel()])
        t_eval = np.linspace(0, self.p.T, self.p.steps + 1)
        
        sol = solve_ivp(self.rhs, (0, self.p.T), y0, method='RK45', 
                        t_eval=t_eval, rtol=1e-4, atol=1e-7)
        return sol