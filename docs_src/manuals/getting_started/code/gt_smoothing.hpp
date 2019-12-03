#pragma once

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>
#include <gridtools/storage/builder.hpp>

using namespace gridtools;
using namespace gridtools::expressions;

#ifdef __CUDACC__
#include <gridtools/stencil_composition/backend/cuda.hpp>
#include <gridtools/storage/cuda.hpp>
using backend_t = cuda::backend<>;
using storage_traits_t = storage::cuda;
#else
#include <gridtools/stencil_composition/backend/mc.hpp>
#include <gridtools/storage/mc.hpp>
using backend_t = mc::backend;
using storage_traits_t = storage::mc;
#endif

static constexpr unsigned halo_size = 2;

constexpr static gridtools::dimension<1> i;
constexpr static gridtools::dimension<2> j;
constexpr static gridtools::dimension<3> k;

using axis_t = axis<2>;
using lower_domain = axis_t::get_interval<0>;
using upper_domain = axis_t::get_interval<1>;

struct lap_function {
    using in = in_accessor<0, extent<-1, 1, -1, 1>>;
    using lap = inout_accessor<1>;

    using param_list = make_param_list<in, lap>;

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

    const auto make_storage = storage::builder<storage_traits_t>.dimensions(Ni, Nj, Nk).type<double>();

#if defined(VARIANT1)
#include "gt_smoothing_variant1_computation.hpp"
#elif defined(VARIANT2)
#include "gt_smoothing_variant2_computation.hpp"
#elif defined(VARIANT3)
#include "gt_smoothing_variant3_computation.hpp"
#endif
}
