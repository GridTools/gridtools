#pragma once

#include <functional>

#include <gridtools/stencil_composition/frontend/make_grid.hpp>
#include <gridtools/storage/builder.hpp>

#ifdef __CUDACC__
#include <gridtools/storage/cuda.hpp>
using storage_traits_t = gridtools::storage::cuda;
#else
#include <gridtools/storage/mc.hpp>
using storage_traits_t = gridtools::storage::mc;
#endif

using data_store_t = decltype(gridtools::storage::builder<storage_traits_t>.dimensions(0, 0, 0).type<double>().build());

using grid_t = decltype(gridtools::make_grid(0, 0, 0));

struct inputs {
    data_store_t in1;
    data_store_t in2;
};
struct outputs {
    data_store_t out;
};

std::function<void(inputs, outputs)> make_interpolate_stencil(grid_t grid, double weight);
