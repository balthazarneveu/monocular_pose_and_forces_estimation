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

def generate_trajectory(
        initial_position: np.ndarray = np.array([2, 4, 1]), 
        initial_velocity: np.ndarray = np.array([10, 15, 20])
    ) -> np.ndarray:
    times = np.linspace(0, 2, 21)  # e.g., 0, 0.1, 0.2, ..., 2
    trajectory = projectilelib.simulate_trajectory(initial_position, initial_velocity, times)
    return trajectory

def main():
    trajectory = generate_trajectory()
    plt.plot(trajectory, "-+")
    plt.show()
    # print(trajectory)

if __name__ == '__main__':
    main()

