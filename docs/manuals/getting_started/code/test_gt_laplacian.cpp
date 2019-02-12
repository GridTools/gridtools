#include <gridtools/stencil-composition/backend.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/storage/storage-facility.hpp>

#include <gridtools/stencil-composition/accessor.hpp>

using namespace gridtools;
using namespace gridtools::expressions;
using namespace gridtools::enumtype; // TODO we need to fix this!

#ifdef __CUDACC__
using target_t = target::cuda;
#else
using target_t = target::mc;
#endif
using backend_t = backend<target_t, grid_type::structured, strategy::block>;

using storage_info_t = storage_traits<target_t>::storage_info_t<0, 3, halo<1, 1, 0>>;
using data_store_t = storage_traits<target_t>::data_store_t<double, storage_info_t>;

constexpr static gridtools::dimension<1> i;
constexpr static gridtools::dimension<2> j;
constexpr static gridtools::dimension<3> k;

struct lap_function {
    using in = in_accessor<0, extent<-1, 1, -1, 1>>;
    using lap = inout_accessor<1>;

    using param_list = make_param_list<in, lap>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation const &eval) {
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

    using arg_phi = arg<0, data_store_t>;
    using arg_lap = arg<1, data_store_t>;

    int halo_size = 1;
    halo_descriptor boundary_i(halo_size, halo_size, halo_size, Ni - halo_size - 1, Ni);
    halo_descriptor boundary_j(halo_size, halo_size, halo_size, Nj - halo_size - 1, Nj);
    auto my_grid = make_grid(boundary_i, boundary_j, Nk);

    auto laplacian = make_computation<backend_t>(          //
        my_grid,                                           //
        make_multistage(                                   //
            execute<parallel>(),                           //
            make_stage<lap_function>(arg_phi(), arg_lap()) //
            ));                                            //

    laplacian.run(arg_phi{} = phi, arg_lap{} = lap);
} // end marker
