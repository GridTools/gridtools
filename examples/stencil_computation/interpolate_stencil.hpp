#pragma once

#include <gridtools/stencil_composition/computation.hpp>
#include <gridtools/stencil_composition/global_parameter.hpp>
#include <gridtools/stencil_composition/grid.hpp>

#ifdef USE_GPU
using backend_t = gridtools::backend::cuda;
#else
using backend_t = gridtools::backend::mc;
#endif

using storage_info_t = gridtools::storage_traits<backend_t>::storage_info_t<0, 3>;
using data_store_t = gridtools::storage_traits<backend_t>::data_store_t<double, storage_info_t>;

using grid_t = decltype(gridtools::make_grid(0, 0, 0));

struct interpolate_stencil {
  public:
    interpolate_stencil(grid_t const &grid, double weight);

    struct inputs {
        data_store_t in1;
        data_store_t in2;
    };
    struct outputs {
        data_store_t out;
    };

    void run(inputs const &inputs, outputs const &outputs);

  private:
    using p_in1 = gridtools::arg<0, data_store_t>;
    using p_in2 = gridtools::arg<1, data_store_t>;
    using p_weight = gridtools::arg<2, gridtools::global_parameter<backend_t, double>>;
    using p_out = gridtools::arg<3, data_store_t>;

    // the wrapped computation is stored in a `gridtools::computation`, which is type erasued. Compilation of
    // `computation` is much less than `make_computation`.
    gridtools::computation<p_in1, p_in2, p_out> m_stencil;
};
