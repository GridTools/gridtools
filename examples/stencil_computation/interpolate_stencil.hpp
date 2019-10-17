#pragma once

#include <functional>

#include <gridtools/commom/defs.hpp>
#include <gridtools/stencil_composition/grid.hpp>
#include <gridtools/storage/storage_facility.hpp>

#ifdef USE_GPU
using backend_t = gridtools::backend::cuda;
#else
using backend_t = gridtools::backend::mc;
#endif

using storage_info_t = gridtools::storage_traits<backend_t>::storage_info_t<0, 3>;
using data_store_t = gridtools::storage_traits<backend_t>::data_store_t<double, storage_info_t>;

using grid_t = decltype(gridtools::make_grid(0, 0, 0));

struct inputs {
    data_store_t in1;
    data_store_t in2;
};
struct outputs {
    data_store_t out;
};

std::function<void(inputs const &, outputs const &)> make_interpolate_stencil(grid_t grid, double weight);
