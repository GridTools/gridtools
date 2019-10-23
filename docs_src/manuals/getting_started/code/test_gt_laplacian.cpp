#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/storage/storage_facility.hpp>

using namespace gridtools;
using namespace gridtools::expressions;

#if defined(__CUDACC__) || defined(__HIPCC__)
using backend_t = backend::cuda;
#else
using backend_t = backend::mc;
#endif

using storage_info_t = storage_traits<backend_t>::storage_info_t<0, 3, halo<1, 1, 0>>;
using data_store_t = storage_traits<backend_t>::data_store_t<double, storage_info_t>;

constexpr static gridtools::dimension<1> i;
constexpr static gridtools::dimension<2> j;
constexpr static gridtools::dimension<3> k;

struct lap_function {
    using in = in_accessor<0, extent<-1, 1, -1, 1>>;
    using lap = inout_accessor<1>;

    using param_list = make_param_list<in, lap>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation const &eval) {
        eval(lap(i, j, k)) = -4. * eval(in(i, j, k)) //
                             + eval(in(i + 1, j, k)) //
                             + eval(in(i, j + 1, k)) //
                             + eval(in(i - 1, j, k)) //
                             + eval(in(i, j - 1, k));
    }
};

int main() {
    uint_t Ni = 10;
    uint_t Nj = 12;
    uint_t Nk = 20;

    storage_info_t info(Ni, Nj, Nk);

    data_store_t phi(info, -1., "phi");
    data_store_t lap(info, -1., "lap");

    int halo_size = 1;
    halo_descriptor boundary_i(halo_size, halo_size, halo_size, Ni - halo_size - 1, Ni);
    halo_descriptor boundary_j(halo_size, halo_size, halo_size, Nj - halo_size - 1, Nj);
    auto my_grid = make_grid(boundary_i, boundary_j, Nk);

    run([](auto phi, auto lap) { return execute_parallel().stage(lap_function(), phi, lap); },
        backend_t(),
        my_grid,
        phi,
        lap);
} // end marker
