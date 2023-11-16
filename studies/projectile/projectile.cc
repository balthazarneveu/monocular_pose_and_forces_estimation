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
        residual[0] = predicted_position[0] - T(observed_position[0]);
        residual[1] = predicted_position[1] - T(observed_position[1]);
        residual[2] = predicted_position[2] - T(observed_position[2]);
        return true;
    }

private:
    const Eigen::Vector3d observed_position;
    const double time;
};

int main()
{
    // Original initial velocity for simulation
    Eigen::Vector3d original_initial_velocity(10, 15, 20);
    Eigen::Vector3d original_initial_position(2, 4, 1);

    // Generate simulated trajectory
    std::vector<std::pair<double, Eigen::Vector3d>> observations;
    for (double t = 0; t <= 2; t += 0.1)
    { // Sample at every 0.1 seconds
        Eigen::Vector3d simulated_position = simulatePosition(
            original_initial_position,
            original_initial_velocity,
            t);
        observations.emplace_back(t, simulated_position);
    }

    // Initial velocity (to be estimated by the solver)
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
    options.max_num_iterations = 50;             // You can adjust this as needed

    ceres::Solve(options, &problem, &summary);

    // Print a brief report
    std::cout << summary.BriefReport() << "\n";

    std::cout << "Estimated initial velocity: " << estimated_initial_velocity.transpose() << std::endl;
    std::cout << "Estimated initial position: " << estimated_initial_position.transpose() << std::endl;

    return 0;
}
