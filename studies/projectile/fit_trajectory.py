import sys
from pathlib import Path
BUILD_DIR  = (Path(__file__).parent/"build").resolve()

assert BUILD_DIR.exists()
sys.path.append(str(BUILD_DIR))
try:
    import projectilelib
except ImportError as e:
    print(f"Import error {e} - Please build first")

import numpy as np
import matplotlib.pyplot as plt

COLORS = "rgbcmk"
def generate_trajectory(
        initial_position: np.ndarray = np.array([2, 4, 1]), 
        initial_velocity: np.ndarray = np.array([10, 15, 20])
    ) -> np.ndarray:
    times = np.linspace(0, 4, 20)
    trajectory = projectilelib.simulate_trajectory(initial_position, initial_velocity, times)
    return trajectory, times

def main():
    trajectory, times = generate_trajectory()
    estimated_initial_position, estimated_initial_velocity = projectilelib.fit(trajectory, times)
    print(estimated_initial_position, estimated_initial_velocity)
    dt_init_visu = 0.2
    velocity_vector = np.vstack([estimated_initial_position, estimated_initial_position+estimated_initial_velocity*dt_init_visu])
    
    for dim_idx, dim in enumerate(["X", "Y", "Z"]):
        color = COLORS[dim_idx%len(COLORS)]
        plt.plot(times, np.array(trajectory)[:, dim_idx], "-+", alpha=0.5, color=color, label=f"{dim} Groundtruth trajectory ")
        plt.plot(
            [times[0],times[0]+dt_init_visu] , velocity_vector[:, dim_idx], "->", 
            alpha=1, color=color,
            label=f"{dim} Estimated initial velocity {estimated_initial_velocity[dim_idx]:.2f}[m/s]"
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Projectile position [m]")
    plt.grid()
    plt.legend()
    plt.title("CERES projectile fitting")
    plt.show()

if __name__ == '__main__':
    main()

