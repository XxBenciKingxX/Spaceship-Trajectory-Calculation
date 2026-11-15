import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Dimensionless constants
G = 4 * np.pi**2

def compute_accelerations(positions, masses):
    N = len(masses)
    accelerations = np.zeros((N, 3))
    
    # Softening parameter to prevent infinite gravitational acceleration during close encounters
    epsilon = 0.05 

    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec)

                # Modified gravitational formula with softening
                accelerations[i] += (
                    G * masses[j] * r_vec / (r_mag**2 + epsilon**2)**1.5
                )

    return accelerations

def rk4_step(positions, velocities, masses, dt):
    k1_v = compute_accelerations(positions, masses)
    k1_r = velocities

    k2_v = compute_accelerations(
        positions + 0.5 * dt * k1_r, masses
    )
    k2_r = velocities + 0.5 * dt * k1_v

    k3_v = compute_accelerations(
        positions + 0.5 * dt * k2_r, masses
    )
    k3_r = velocities + 0.5 * dt * k2_v

    k4_v = compute_accelerations(
        positions + dt * k3_r, masses
    )
    k4_r = velocities + dt * k3_v

    new_positions = positions + (dt / 6) * (
        k1_r + 2*k2_r + 2*k3_r + k4_r
    )

    new_velocities = velocities + (dt / 6) * (
        k1_v + 2*k2_v + 2*k3_v + k4_v
    )

    return new_positions, new_velocities

def euler_step(positions, velocities, masses, dt):
    accelerations = compute_accelerations(positions, masses)
    new_positions = positions + velocities * dt
    new_velocities = velocities + accelerations * dt
    return new_positions, new_velocities


# Initialization
masses = np.array([1.0, 3.003e-6, 3.213e-7, 0.0])

r1 = 1.0      
r2 = 1.524    
v_earth = np.sqrt(G / r1)
v_mars = np.sqrt(G / r2)

# Hohmann Transfer Calculation
a_transfer = (r1 + r2) / 2.0  
v_spacecraft = np.sqrt(G * (2.0 / r1 - 1.0 / a_transfer)) 

# Phase angle for Mars rendezvous at aphelion
t_transfer = np.pi * np.sqrt(a_transfer**3 / G)
mars_angle = np.pi - (v_mars / r2) * t_transfer

initial_positions = np.array([
    [0.0, 0.0, 0.0],                                             # Sun
    [r1, 0.0, 0.0],                                              # Earth
    [r2 * np.cos(mars_angle), r2 * np.sin(mars_angle), 0.0],     # Mars
    [r1, 0.0, 0.0]                                               # Spaceship starts at Earth
])

initial_velocities = np.array([
    [0.0, 0.0, 0.0],
    [0.0, v_earth, 0.0],
    [-v_mars * np.sin(mars_angle), v_mars * np.cos(mars_angle), 0.0],
    [0.0, v_spacecraft, 0.0]
])

positions_rk4 = initial_positions.copy()
velocities_rk4 = initial_velocities.copy()

positions_euler = initial_positions.copy()
velocities_euler = initial_velocities.copy()

# Simulation Parameters
dt = 0.001
steps = 3000  

history_rk4 = np.zeros((steps, 4, 3))
history_euler = np.zeros((steps, 4, 3))

print("Simulation started...")

for step in range(steps):
    history_rk4[step] = positions_rk4
    history_euler[step] = positions_euler
    
    positions_rk4, velocities_rk4 = rk4_step(positions_rk4, velocities_rk4, masses, dt)
    positions_euler, velocities_euler = euler_step(positions_euler, velocities_euler, masses, dt)

print("Simulation finished. Generating plot...")

# --- 3D VISUALIZATION ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Background styling
ax.set_facecolor('#05070e')
fig.patch.set_facecolor('#05070e')
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor('#1f2937')

# Plot Sun
ax.plot([history_rk4[0, 0, 0]], [history_rk4[0, 0, 1]], [history_rk4[0, 0, 2]], 
        'yo', markersize=14, label='Sun')
ax.plot([history_rk4[0, 0, 0]], [history_rk4[0, 0, 1]], [history_rk4[0, 0, 2]], 
        'yo', markersize=24, alpha=0.25)

# Faint static reference orbits
ax.plot(history_rk4[:, 1, 0], history_rk4[:, 1, 1], history_rk4[:, 1, 2], 'b-', alpha=0.15, lw=1)
ax.plot(history_rk4[:, 2, 0], history_rk4[:, 2, 1], history_rk4[:, 2, 2], 'r-', alpha=0.15, lw=1)

# Dynamic trail lines and body markers
earth_trail, = ax.plot([], [], [], 'b-', lw=1.8, label='Earth')
earth_point, = ax.plot([], [], [], 'bo', markersize=7)

mars_trail, = ax.plot([], [], [], 'r-', lw=1.8, label='Mars')
mars_point, = ax.plot([], [], [], 'ro', markersize=6)

ship_rk4_trail, = ax.plot([], [], [], 'g--', lw=2.0, label='Spaceship (RK4)')
ship_rk4_point, = ax.plot([], [], [], 'go', markersize=5)

ship_euler_trail, = ax.plot([], [], [], 'm:', lw=1.5, alpha=0.8, label='Spaceship (Euler)')
ship_euler_point, = ax.plot([], [], [], 'm^', markersize=5)

# Time and distance text display
info_text = ax.text2D(0.03, 0.95, '', transform=ax.transAxes, family='monospace', fontsize=10,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#111827', alpha=0.8))

# Axis limits and labels
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_zlim(-1.0, 1.0)
ax.set_title('Orbital Simulation: Spaceship Transfer (3D)')
ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.set_zlabel('z (AU)')
ax.legend(loc='upper right')
ax.set_box_aspect([1, 1, 1])

# Animation configuration
tail_length = 400
step_skip = 15

def update(frame):
    i = min(frame * step_skip, steps - 1)
    start = max(0, i - tail_length)

    # Update Earth
    earth_trail.set_data(history_rk4[start:i, 1, 0], history_rk4[start:i, 1, 1])
    earth_trail.set_3d_properties(history_rk4[start:i, 1, 2])
    earth_point.set_data([history_rk4[i, 1, 0]], [history_rk4[i, 1, 1]])
    earth_point.set_3d_properties([history_rk4[i, 1, 2]])

    # Update Mars
    mars_trail.set_data(history_rk4[start:i, 2, 0], history_rk4[start:i, 2, 1])
    mars_trail.set_3d_properties(history_rk4[start:i, 2, 2])
    mars_point.set_data([history_rk4[i, 2, 0]], [history_rk4[i, 2, 1]])
    mars_point.set_3d_properties([history_rk4[i, 2, 2]])

    # Update Spaceship (RK4)
    ship_rk4_trail.set_data(history_rk4[:i, 3, 0], history_rk4[:i, 3, 1])
    ship_rk4_trail.set_3d_properties(history_rk4[:i, 3, 2])
    ship_rk4_point.set_data([history_rk4[i, 3, 0]], [history_rk4[i, 3, 1]])
    ship_rk4_point.set_3d_properties([history_rk4[i, 3, 2]])

    # Update Spaceship (Euler)
    ship_euler_trail.set_data(history_euler[:i, 3, 0], history_euler[:i, 3, 1])
    ship_euler_trail.set_3d_properties(history_euler[:i, 3, 2])
    ship_euler_point.set_data([history_euler[i, 3, 0]], [history_euler[i, 3, 1]])
    ship_euler_point.set_3d_properties([history_euler[i, 3, 2]])

    # Update telemetry text
    t_years = i * dt
    dist_to_mars = np.linalg.norm(history_rk4[i, 3] - history_rk4[i, 2])
    euler_error = np.linalg.norm(history_rk4[i, 3] - history_euler[i, 3])
    info_text.set_text(
        f"Step        : {i}/{steps}\n"
        f"Time        : {t_years:.3f} yr\n"
        f"Mars Dist   : {dist_to_mars:.3f} AU\n"
        f"Euler Error : {euler_error:.3f} AU"
    )

    # Rotate camera slowly
    ax.view_init(elev=25, azim=-60 + frame * 0.3)

    return (earth_trail, earth_point, mars_trail, mars_point,
            ship_rk4_trail, ship_rk4_point, ship_euler_trail, ship_euler_point, info_text)

ani = FuncAnimation(fig, update, frames=steps // step_skip, interval=30, blit=False)

plt.show(block=True)