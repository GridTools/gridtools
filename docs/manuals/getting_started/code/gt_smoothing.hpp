#include <gridtools/stencil-composition/backend.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/stencil-composition/stencil-functions/stencil-functions.hpp>
#include <gridtools/storage/storage-facility.hpp>

using namespace gridtools;
using namespace gridtools::expressions;
using namespace gridtools::enumtype; // TODO we need to fix this!

#ifdef __CUDACC__
using target_t = target::cuda;
#else
using target_t = target::mc;
#endif
using backend_t = backend<target_t, grid_type::structured, strategy::block>;

static constexpr unsigned halo_size = 2;

using storage_info_t = storage_traits<target_t>::storage_info_t<0, 3, halo<halo_size, halo_size, 0>>;
using data_store_t = storage_traits<target_t>::data_store_t<double, storage_info_t>;

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

    storage_info_t info(Ni, Nj, Nk);

#if defined(VARIANT1)
#include "gt_smoothing_variant1_computation.hpp"
#elif defined(VARIANT2)
#include "gt_smoothing_variant2_computation.hpp"
#elif defined(VARIANT3)
#include "gt_smoothing_variant3_computation.hpp"
#endif
}
