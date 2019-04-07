#pragma once

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>
#include <gridtools/storage/storage_facility.hpp>

using namespace gridtools;
using namespace gridtools::expressions;

#ifdef __CUDACC__
using backend_t = backend::cuda;
#else
using backend_t = backend::mc;
#endif

static constexpr unsigned halo_size = 2;

using storage_info_t = storage_traits<backend_t>::storage_info_t<0, 3, halo<halo_size, halo_size, 0>>;
using data_store_t = storage_traits<backend_t>::data_store_t<double, storage_info_t>;

constexpr static gridtools::dimension<1> i;
constexpr static gridtools::dimension<2> j;
constexpr static gridtools::dimension<3> k;

using axis_t = axis<2>;
using lower_domain = axis_t::get_interval<0>;
using upper_domain = axis_t::get_interval<1>;

struct lap_function {
    GT_DEFINE_ACCESSORS(GT_IN_ACCESSOR(in, extent<-1, 1, -1, 1>), GT_INOUT_ACCESSOR(lap));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(lap(i, j, k)) = -4. * eval(in(i, j, k)) //
                             + eval(in(i + 1, j, k)) //
                             + eval(in(i, j + 1, k)) //
                             + eval(in(i - 1, j, k)) //
                             + eval(in(i, j - 1, k));
    }
};

#if defined(VARIANT1) || defined(VARIANT2)
#include "gt_smoothing_variant1_operator.hpp"
#elif defined(VARIANT3)
#include "gt_smoothing_variant3_operator.hpp"
#endif

int main() {
    uint_t Ni = 50;
    uint_t Nj = 50;
    uint_t Nk = 20;
    uint_t kmax = 12;

    storage_info_t info(Ni, Nj, Nk);

#if defined(VARIANT1)
#include "gt_smoothing_variant1_computation.hpp"
#elif defined(VARIANT2)
#include "gt_smoothing_variant2_computation.hpp"
#elif defined(VARIANT3)
#include "gt_smoothing_variant3_computation.hpp"
#endif
}
