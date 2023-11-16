#include <vector>
#include <ceres/ceres.h>
#include <Eigen/Core>


// Gravity (m/s^2)
const Eigen::Vector3d gravity(0, 0, -9.81);

// Function to simulate position
Eigen::Vector3d simulatePosition(const Eigen::Vector3d &initial_position, const Eigen::Vector3d &initial_velocity, double time)
{
    return 0.5 * gravity * time * time + initial_velocity * time + initial_position;
}
