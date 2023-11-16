cmake_minimum_required(VERSION 3.5)

project(projectile)

find_package(Ceres REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(projectilelib projectile.cpp)
target_link_libraries(projectilelib PRIVATE pybind11::pybind11 Ceres::ceres)


# CERES TOY EXAMPLE
add_executable(projectile projectile.cc)
target_link_libraries(projectile Ceres::ceres)
