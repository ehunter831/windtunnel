import numpy as np
from typing import Tuple

class LBMSolver:
    """
    D2Q9 BGK LBM solver (lattice units).
    Notes:
    - c_s^2 = 1/3
    - nu = c_s^2 * (tau - 0.5)
    - Use lattice units: choose U_char and L_char in lattice units to compute tau from desired Re.
    """
    def __init__(self, nx: int, ny: int, obstacle_mask: np.ndarray,
                 tau: float = 0.6, inlet_velocity: float = 0.1, dtype=np.float64):
        self.nx = nx
        self.ny = ny
        self.mask = obstacle_mask.astype(bool)
        self.dtype = dtype
        self.inlet_velocity = inlet_velocity

        # D2Q9
        self.cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=int)
        self.cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=int)
        self.weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=dtype)
        self.opposites = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)

        self.tau = float(tau)
        self.cs2 = 1.0 / 3.0
        self.nu = self.cs2 * (self.tau - 0.5)

        # distributions (9, ny, nx) - initialize with inlet velocity
        self.F = np.zeros((9, ny, nx), dtype=dtype)
        rho_init = 1.0
        ux_init = inlet_velocity
        uy_init = 0.0
        u_sq = ux_init**2 + uy_init**2
        for i, w in enumerate(self.weights):
            cu = self.cxs[i] * ux_init + self.cys[i] * uy_init
            self.F[i, :, :] = rho_init * w * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)

    def macroscopic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = np.sum(self.F, axis=0)
        # avoid division by zero
        rho_safe = np.where(rho <= 0, 1.0, rho)
        ux = np.sum(self.F * self.cxs[:, None, None], axis=0) / rho_safe
        uy = np.sum(self.F * self.cys[:, None, None], axis=0) / rho_safe
        # force zero velocity inside solids
        ux[self.mask] = 0.0
        uy[self.mask] = 0.0
        return rho, ux, uy

    def step(self):
        rho, ux, uy = self.macroscopic()
        # compute equilibrium and collision (BGK)
        u_sq = ux**2 + uy**2
        F_eq = np.zeros_like(self.F)
        for i, w in enumerate(self.weights):
            cu = self.cxs[i] * ux + self.cys[i] * uy
            F_eq[i] = rho * w * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)
        # collision
        omega = 1.0 / self.tau
        self.F += omega * (F_eq - self.F)

        # streaming
        for i in range(9):
            self.F[i] = np.roll(self.F[i], shift=self.cxs[i], axis=1)
            self.F[i] = np.roll(self.F[i], shift=self.cys[i], axis=0)

        # bounce-back for solid nodes (post-streaming, simple/full-node)
        solid = self.mask
        for i in range(1, 9):
            self.F[self.opposites[i]][solid] = self.F[i][solid]

        # Inlet boundary condition (left side, x=0) - constant velocity
        rho_inlet = 1.0
        ux_inlet = self.inlet_velocity
        uy_inlet = 0.0
        u_sq_inlet = ux_inlet**2 + uy_inlet**2
        for i, w in enumerate(self.weights):
            cu = self.cxs[i] * ux_inlet + self.cys[i] * uy_inlet
            self.F[i, :, 0] = rho_inlet * w * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq_inlet)

        # Outlet boundary condition (right side, x=nx-1) - open/outflow
        # Simple copy from previous column
        self.F[:, :, -1] = self.F[:, :, -2]

        # return velocities for visualization
        _, ux, uy = self.macroscopic()
        return ux, uy
    
def create_grid(nx, ny):
    #create the mesh grid
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y

def create_obstacle_mask(X, Y, center_x, center_y, radius):
    #true or false grid representing the obstacle
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = dist_from_center < radius
    return mask
