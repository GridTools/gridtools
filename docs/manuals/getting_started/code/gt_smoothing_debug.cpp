#include <gridtools/stencil-composition/backend.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/storage/storage-facility.hpp>

using namespace gridtools;

using target_t = target::x86;
using backend_t = backend<target_t, grid_type::structured, strategy::naive>;

using storage_info_t = storage_traits<target_t>::storage_info_t<0, 3>;
using data_store_t = storage_traits<target_t>::data_store_t<double, storage_info_t>;

struct lap_function {
    using in = in_accessor<0, extent<-1, 1, -1, 1>>;
    using lap = inout_accessor<1>;

    using arg_list = boost::mpl::vector<in, lap>;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(lap()) = -4. * eval(in())     //
                      + eval(in(1, 0, 0))  //
                      + eval(in(0, 1, 0))  //
                      + eval(in(-1, 0, 0)) //
                      + eval(in(0, -1, 0));
    }
};

struct smoothing_function_1 {
    using phi = in_accessor<0>;
    using laplap = in_accessor<1>;
    using out = inout_accessor<2>;

    using arg_list = boost::mpl::vector<phi, laplap, out>;

    constexpr static double alpha = 0.5;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval) {
        eval(out()) = eval(phi()) //
                      - alpha * eval(laplap());
    }
};

int main() {
    int Ni = 50;
    int Nj = 50;
    int Nk = 20;
    int kmax = 12;

    storage_info_t info(Ni, Nj, Nk);

    data_store_t phi(info);
    data_store_t phi_new(info);

    using arg_phi = arg<0, data_store_t>;
    using arg_phi_new = arg<1, data_store_t>;
    using arg_lap = tmp_arg<2, data_store_t>;
    using arg_laplap = tmp_arg<3, data_store_t>;

    int bs = 2; // boundary size
    halo_descriptor boundary_i(bs, bs, bs, Ni - bs - 1, Ni);
    halo_descriptor boundary_j(bs, bs, bs, Nj - bs - 1, Nj);
    auto my_grid = make_grid(boundary_i, boundary_j, Nk);

    auto smoothing = make_computation<backend_t>(              /***/
        my_grid,                                               /***/
        make_multistage(                                       /***/
            enumtype::execute<enumtype::forward>(),            /***/
            make_stage<lap_function>(arg_phi(), arg_lap()),    /***/
            make_stage<lap_function>(arg_lap(), arg_laplap()), /***/
            make_stage<smoothing_function_1>(                  /***/
                arg_phi(),                                     /***/
                arg_laplap(),                                  /***/
                arg_phi_new())                                 /***/
            ));                                                /***/

    smoothing.run(arg_phi{} = phi, arg_phi_new{} = phi_new);
}
