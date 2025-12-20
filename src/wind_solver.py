import numpy as np

class LBMSolver:
    def __init__(self, nx, ny, obstacle_mask):
        self.nx = nx
        self.ny = ny
        self.mask = obstacle_mask
        
        # --- LBM CONSTANTS (D2Q9 Model) ---
        # The 9 directions: Center, E, N, W, S, NE, NW, SW, SE
        self.idxs = np.arange(9)
        self.cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        self.weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        
        # Bounce-back indices (If moving Right (1), bounce acts as Left (3))
        self.opposites = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        # --- SIMULATION PARAMETERS ---
        self.tau = 0.6  # Relaxation time (Controls viscosity. Lower = thinner fluid)
        # Omega = 1 / tau. Higher omega = more turbulence, but less stable.
        
        # --- INITIALIZATION ---
        # F is the "Distribution Function" (9 layers of grids)
        # We start with a slight flow to the right
        self.F = np.zeros((9, ny, nx)) 
        for i, w in enumerate(self.weights):
            self.F[i] =  w * 1.0

    def step(self):
        # 1. MACROSCOPIC VARIABLES (Density & Velocity)
        rho = np.sum(self.F, axis=0) # Sum of all 9 directions = Density
        ux = np.sum(self.F * self.cxs[:, None, None], axis=0) / rho
        uy = np.sum(self.F * self.cys[:, None, None], axis=0) / rho
        
        # 2. COLLISION (The Physics)
        # Calculate Equilibrium (Where the fluid WANTS to be)
        # This formula is the heart of LBM (Navier-Stokes approximation)
        u_sq = ux**2 + uy**2
        F_eq = np.zeros_like(self.F)
        for i, w in enumerate(self.weights):
            cu = self.cxs[i]*ux + self.cys[i]*uy
            F_eq[i] = rho * w * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)
        
        # Relax F towards F_eq
        self.F += -(1.0 / self.tau) * (self.F - F_eq)
        
        # 3. BOUNDARY CONDITIONS (Obstacles)
        # "Bounce Back": If you hit a wall, reflect the data backward
        # We extract the fluid going INTO the wall, and force it back OUT
        # (This is a simplified bounce-back for clarity)
        boundary_indices = np.where(self.mask)
        for i in range(1, 9):
            self.F[self.opposites[i]][boundary_indices] = self.F[i][boundary_indices]

        # 4. STREAMING (Movement)
        for i in range(9):
            self.F[i] = np.roll(self.F[i], self.cxs[i], axis=1)
            self.F[i] = np.roll(self.F[i], self.cys[i], axis=0)
        rho_inlet = 1.0
        ux_inlet = 0.1 
        uy_inlet = 0.0
        
        u_sq_inlet = ux_inlet**2 + uy_inlet**2

        for i, w in enumerate(self.weights):
            cu = self.cxs[i]*ux_inlet + self.cys[i]*uy_inlet
            inlet_eq = rho_inlet * w * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq_inlet)
            self.F[i, :, 0] = inlet_eq
        
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
