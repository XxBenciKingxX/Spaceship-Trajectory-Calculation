import numpy as np
import matplotlib.pyplot as plt

# Dimensionless constants
G = 4 * np.pi**2

def compute_accelerations(positions, masses):
    N = len(masses)
    accelerations = np.zeros((N, 3))

    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec)

                # Avoid division by zero
                if r_mag < 1e-10:
                    continue

                accelerations[i] += (
                    G * masses[j] * r_vec / r_mag**3
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

initial_positions = np.array([
    [0.0, 0.0, 0.0],       # Sun
    [1.0, 0.0, 0.0],       # Earth
    [1.524, 0.0, 0.0],     # Mars
    [1.0, 0.0001, 0.0]     # Spaceship, slightly offset
])

v_earth = np.sqrt(G / 1.0)
v_mars = np.sqrt(G / 1.524)

# Hohmann Transfer Calculation for the Spaceship to reach Mars
r1 = 1.0      
r2 = 1.524    
a_transfer = (r1 + r2) / 2.0  
v_spacecraft = np.sqrt(G * (2.0 / r1 - 1.0 / a_transfer)) 

initial_velocities = np.array([
    [0.0, 0.0, 0.0],
    [0.0, v_earth, 0.0],
    [0.0, v_mars, 0.0],
    [0.0, v_spacecraft, 0.0]
])

# Initialize separate state arrays for RK4 and Euler tracking
positions_rk4 = initial_positions.copy()
velocities_rk4 = initial_velocities.copy()

positions_euler = initial_positions.copy()
velocities_euler = initial_velocities.copy()

# Simulation Parameters
dt = 0.001
steps = 3000  # Increased to 3 simulated years to guarantee a full Mars orbit

history_rk4 = np.zeros((steps, 4, 3))
history_euler = np.zeros((steps, 4, 3))

print("Simulation started...")

# Simulation Loop
for step in range(steps):
    history_rk4[step] = positions_rk4
    history_euler[step] = positions_euler
    
    positions_rk4, velocities_rk4 = rk4_step(positions_rk4, velocities_rk4, masses, dt)
    positions_euler, velocities_euler = euler_step(positions_euler, velocities_euler, masses, dt)

print("Simulation finished. Generating 3D plot...")

# --- 3D VISUALIZATION ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the Sun as a single static marker in 3D
ax.plot([history_rk4[0, 0, 0]], [history_rk4[0, 0, 1]], [history_rk4[0, 0, 2]], 
        'yo', markersize=15, label='Sun')

# Plot continuous paths for the moving bodies in 3D
ax.plot(history_rk4[:, 1, 0], history_rk4[:, 1, 1], history_rk4[:, 1, 2], 'b-', label='Earth')
ax.plot(history_rk4[:, 2, 0], history_rk4[:, 2, 1], history_rk4[:, 2, 2], 'r-', label='Mars')
ax.plot(history_rk4[:, 3, 0], history_rk4[:, 3, 1], history_rk4[:, 3, 2], 'g--', label='Spaceship', linestyle='-')

ax.set_title('Orbital Simulation: Spaceship Transfer (3D)')
ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ax.set_zlabel('z (AU)')
ax.legend()

# Set aspect ratio to be equal so orbits don't look stretched
ax.set_box_aspect([1, 1, 1])

plt.show(block=True)