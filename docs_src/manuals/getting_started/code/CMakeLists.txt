cmake_minimum_required(VERSION 3.12.4)

project(GridTools-laplacian LANGUAGES CXX)

find_package(GridTools 1.1.0 REQUIRED)

add_executable(gt_laplacian test_gt_laplacian.cpp)
target_link_libraries(gt_laplacian GridTools::gridtools)
