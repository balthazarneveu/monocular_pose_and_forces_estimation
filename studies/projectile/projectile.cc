#include <vector>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include "simulate.h"

// Cost function for Ceres Solver
struct TrajectoryCostFunctor
{
    TrajectoryCostFunctor(const Eigen::Vector3d &observed_position, double time)
        : observed_position(observed_position), time(time) {}

    template <typename T>
    bool operator()(const T *const initial_position, const T *const initial_velocity, T *residual) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> x0(initial_position);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v0(initial_velocity);
        Eigen::Matrix<T, 3, 1> predicted_position = T(0.5) * gravity.cast<T>() * T(time) * T(time) + v0 * T(time) + x0;
        // F = m*a
        // a = dv/dt -> v = g*t + v0 = dx/dt -> x= F/m *t²/2 + v0*t + x0
        // F= m.g for gravity 
        // => x=gt²/2 + v0.t + x0
        // -> parabola trajectory, does not depend on mass.
        residual[0] = predicted_position[0] - T(observed_position[0]);
        residual[1] = predicted_position[1] - T(observed_position[1]);
        residual[2] = predicted_position[2] - T(observed_position[2]);
        return true;
    }

private:
    const Eigen::Vector3d observed_position;
    const double time;
};



std::pair<Eigen::Vector3d, Eigen::Vector3d> fit(const std::vector<std::pair<double, Eigen::Vector3d>>& observations) {
    // Initial velocity to zero (to be estimated by the solver)
    Eigen::Vector3d estimated_initial_velocity(0, 0, 0);
    Eigen::Vector3d estimated_initial_position(0, 0, 0);

    // Set up the problem.
    ceres::Problem problem;

    for (const auto &obs : observations)
    {
        ceres::CostFunction *cost_function =
            new ceres::AutoDiffCostFunction<TrajectoryCostFunctor, 3, 3, 3>(
                new TrajectoryCostFunctor(obs.second, obs.first));
        problem.AddResidualBlock(cost_function, nullptr, estimated_initial_position.data(), estimated_initial_velocity.data());
    }

    // Run the solver
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    options.minimizer_progress_to_stdout = true; // Enable logging to stdout
    options.max_num_iterations = 50;             // Tuning

    ceres::Solve(options, &problem, &summary);

    // Print a brief report
    std::cout << summary.BriefReport() << "\n";
    // Logging convergence metrics
    std::cout << "Estimated initial velocity: " << estimated_initial_velocity.transpose() << std::endl;
    std::cout << "Estimated initial position: " << estimated_initial_position.transpose() << std::endl;

    
    return {estimated_initial_position, estimated_initial_velocity};
}


int main() {
    // Define initial conditions for the simulation
    Eigen::Vector3d initial_position(0, 0, 0); // Starting at the origin
    Eigen::Vector3d initial_velocity(10, 10, 10); // Some initial velocity

    // Simulate the trajectory
    std::vector<std::pair<double, Eigen::Vector3d>> trajectory;
    for (double t = 0; t <= 2; t += 0.1) { // Simulate for 2 seconds
        Eigen::Vector3d position = simulatePosition(initial_position, initial_velocity, t);
        trajectory.emplace_back(t, position);
    }

    // Fit the trajectory to estimate initial position and velocity
    auto [estimated_position, estimated_velocity] = fit(trajectory);

    return 0;
}