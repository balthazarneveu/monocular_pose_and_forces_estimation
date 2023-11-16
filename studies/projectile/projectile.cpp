#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <vector>
#include "projectile.cc"

namespace py = pybind11;


PYBIND11_MODULE(projectilelib, m) {
    m.doc() = "Binding projectile"; // Optional module docstring

    m.def("simulate_trajectory", [](const Eigen::Vector3d &initial_position, 
                                    const Eigen::Vector3d &initial_velocity,
                                    const std::vector<double> &times) {
        std::vector<Eigen::Vector3d> trajectory;
        for (auto t : times) {
            trajectory.push_back(simulatePosition(initial_position, initial_velocity, t));
        }
        return trajectory;
    }, "Simulate a trajectory given initial position, velocity and times");
}