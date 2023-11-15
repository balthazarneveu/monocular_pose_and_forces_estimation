cmake_minimum_required(VERSION 3.5)

project(projectile)

find_package(Ceres REQUIRED)

# CERES TOY EXAMPLE
add_executable(projectile projectile.cc)
target_link_libraries(projectile Ceres::ceres)