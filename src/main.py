import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import wind_solver as physics

Nx = 300  #width
Ny = 150  #height
num_particles = 4000


X, Y = physics.create_grid(Nx,Ny)

mask = physics.create_obstacle_mask(X, Y, center_x=Nx//2.5, center_y=Ny//2, radius=15)

sim = physics.LBMSolver(Nx, Ny, mask)

particles_x = np.random.uniform(0, Nx, num_particles)
particles_y = np.random.uniform(0, Ny, num_particles)

fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, Nx)
ax.set_ylim(0, Ny)
ax.contourf(X, Y, mask, levels=[0.5, 1], colors='black')
curl_plot = ax.imshow(np.zeros((Ny, Nx)), cmap='bwr', origin='lower', 
                      vmin=-0.05, vmax=0.05, alpha=0.8)

points, = ax.plot(particles_x, particles_y, 'k.', markersize=2, alpha=0.5)

def update(frame):
    global particles_x, particles_y

    for _ in range(3): 
        ux, uy = sim.step()

    dy_ux, dx_ux = np.gradient(ux)
    dy_uy, dx_uy = np.gradient(uy)
    curl = dx_uy - dy_ux
    curl[mask]=np.nan
    curl_plot.set_data(curl)

    ix = np.clip(particles_x, 0, Nx-1).astype(int)
    iy = np.clip(particles_y, 0, Ny-1).astype(int)

    particles_x += ux[iy, ix] * 2.0
    particles_y += uy[iy, ix] * 2.0

    ix = np.clip(particles_x, 0, Nx-1).astype(int)
    iy = np.clip(particles_y, 0, Ny-1).astype(int)
    
    in_obstacle = mask[iy, ix]
    off_screen = particles_x > Nx
    reset_indices = np.logical_or(off_screen, in_obstacle)
    
    particles_x[reset_indices] = 0
    particles_y[reset_indices] = np.random.uniform(0, Ny, np.sum(reset_indices))
    
    points.set_data(particles_x, particles_y)
    return [curl_plot, points]

anim = FuncAnimation(fig, update, frames=500, interval=1, blit=True)
plt.show()