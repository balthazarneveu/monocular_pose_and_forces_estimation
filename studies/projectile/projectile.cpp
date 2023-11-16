#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <vector>
#include "projectile.cc"

namespace py = pybind11;

PYBIND11_MODULE(projectilelib, m)
{
    m.doc() = "Binding projectile"; // Optional module docstring

    m.def(
        "simulate_trajectory", [](const Eigen::Vector3d &initial_position, const Eigen::Vector3d &initial_velocity, const std::vector<double> &times)
        {
        std::vector<Eigen::Vector3d> trajectory;
        for (auto t : times) {
            trajectory.push_back(simulatePosition(initial_position, initial_velocity, t));
        }
        return trajectory; },
        "Simulate a trajectory given initial position, velocity and times");
    m.def(
        "fit", [](const std::vector<Eigen::Vector3d> &trajectory, const std::vector<double> &times)
        {
        if (trajectory.size() != times.size()) {
            throw std::runtime_error("The sizes of trajectory and times must match.");
        }

        std::vector<std::pair<double, Eigen::Vector3d>> observations;
        for (size_t i = 0; i < trajectory.size(); ++i) {
            observations.emplace_back(times[i], trajectory[i]);
        }

        return fit(observations); },
        "Fit initial position and velocity given a trajectory and corresponding times");
}