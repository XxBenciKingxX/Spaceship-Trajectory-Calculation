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
    # 1. Calculate current accelerations based on current positions
    accelerations = compute_accelerations(positions, masses)
    
    # 2. Update positions using current velocities
    new_positions = positions + velocities * dt
    
    # 3. Update velocities using current accelerations
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
v_spacecraft = v_earth + 0.6

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
steps = 1500  # Roughly 1.5 simulated years


history_rk4 = np.zeros((steps, 4, 3))

# Simulation Loop
for step in range(steps):
    history_rk4[step] = positions_rk4
    positions_rk4, velocities_rk4 = rk4_step(positions_rk4, velocities_rk4, masses, dt)
    positions_euler, velocities_euler = euler_step(positions_euler, velocities_euler, masses, dt)

# Visualization
plt.figure(figsize=(10, 8))

# 1. Plot the Sun as a single static marker (only its initial position)
plt.plot(history_rk4[0, 0, 0], history_rk4[0, 0, 1], 'yo', markersize=15, label='Sun}')

# 2. Plot continuous paths for the moving bodies
plt.plot(history_rk4[:, 1, 0], history_rk4[:, 1, 1], 'b-', label='Earth Path')
plt.plot(history_rk4[:, 2, 0], history_rk4[:, 2, 1], 'r-', label='Mars Path')
plt.plot(history_rk4[:, 3, 0], history_rk4[:, 3, 1], 'g--', label='Spaceship Path')

plt.title('Orbital Simulation: Spaceship Transfer')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.legend()
plt.grid(True)
plt.axis('equal') # Ensures circular orbits aren't visually distorted
plt.show()