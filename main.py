
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


# Initialization
masses = np.array([1.0, 3.003e-6, 3.213e-7, 0.0])

positions = np.array([
    [0.0, 0.0, 0.0],       # Sun
    [1.0, 0.0, 0.0],       # Earth
    [1.524, 0.0, 0.0],     # Mars
    [1.0, 0.0001, 0.0]     # Spaceship, slightly offset
])

v_earth = np.sqrt(G / 1.0)
v_mars = np.sqrt(G / 1.524)

v_spacecraft = v_earth + 0.6

velocities = np.array([
    [0.0, 0.0, 0.0],
    [0.0, v_earth, 0.0],
    [0.0, v_mars, 0.0],
    [0.0, v_spacecraft, 0.0]
])

# Simulation parameters
dt = 0.001
total_time = 1.0
num_steps = int(total_time / dt)

history_pos = np.zeros((num_steps, len(masses), 3))

# Simulation loop
for step in range(num_steps):
    history_pos[step] = positions
    positions, velocities = rk4_step(
        positions, velocities, masses, dt
    )


# 3D Visualization
def plot_3d_trajectory(history_pos):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    labels = ['Sun', 'Earth', 'Mars', 'Spaceship']
    colors = ['orange', 'blue', 'red', 'green']
    sizes = [300, 100, 80, 50]

    for i in range(len(labels)):
        x = history_pos[:, i, 0]
        y = history_pos[:, i, 1]
        z = history_pos[:, i, 2]

        ax.plot(
            x, y, z,
            label=f'{labels[i]} Path',
            color=colors[i]
        )

        ax.scatter(
            x[-1], y[-1], z[-1],
            color=colors[i],
            s=sizes[i],
            edgecolors='black'
        )

    ax.set_title("Earth to Mars Transfer")
    ax.set_xlabel("X Position (AU)")
    ax.set_ylabel("Y Position (AU)")
    ax.set_zlabel("Z Position (AU)")

    ax.legend()
    plt.tight_layout()
    plt.show()


plot_3d_trajectory(history_pos)
#csao
