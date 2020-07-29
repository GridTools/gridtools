cmake_minimum_required(VERSION 3.14.5)

project(GridTools-laplacian LANGUAGES CXX)

find_package(GridTools 2.0.0 REQUIRED QUIET)

if(TARGET GridTools::stencil_cpu_ifirst)
    add_executable(gt_laplacian test_gt_laplacian.cpp)
    target_link_libraries(gt_laplacian GridTools::gridtools GridTools::stencil_cpu_ifirst)
endif()
