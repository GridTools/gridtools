#include <gridtools/stencil-composition/stencil-composition.hpp>

using namespace gridtools;

struct smoothing_function_1 {
    using phi = in_accessor<0>;
    using laplap = in_accessor<1>;
    using out = inout_accessor<2>;

    using arg_list = boost::mpl::vector<phi, laplap, out>;

    constexpr static double alpha = 0.5;

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation const &eval, lower_domain) {
        eval(out(i, j, k)) = eval(phi(i, j, k)) //
                             - alpha * eval(laplap(i, j, k));
    }

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation const &eval, upper_domain) {
        eval(out(i, j, k)) = eval(phi(i, j, k));
    }
};

using axis_t = axis<2>;
using lower_domain = axis_t::get_interval<0>;
using upper_domain = axis_t::get_interval<1>;
using full_domain = axis_t::full_interval;

int main() {
    data_store_t phi(info);
    data_store_t phi_new(info);
    data_store_t lap(info);
    data_store_t laplap(info);

    using arg_phi = arg<0, data_store_t>;
    using arg_phi_new = arg<1, data_store_t>;
    using arg_lap = arg<2, data_store_t>;
    using arg_laplap = arg<3, data_store_t>;

    using arg_list = boost::mpl::vector<arg_phi, arg_phi_new, arg_lap, arg_laplap>;
    aggregator_type<arg_list> domain(phi, phi_new, lap, laplap);

    int bs = 2; // boundary size
    halo_descriptor boundary_i(bs, bs, bs, Ni - bs - 1, Ni);
    halo_descriptor boundary_j(bs, bs, bs, Nj - bs - 1, Nj);
    auto my_grid = make_grid(boundary_i, boundary_j, axis_t{kmax, Nk - kmax});

    auto smoothing = make_computation<backend_t>(              //
        domain,                                                //
        my_grid,                                               //
        make_multistage(                                       //
            execute<forward>(),                                //
            make_stage<lap_function>(arg_phi(), arg_lap()),    //
            make_stage<lap_function>(arg_lap(), arg_laplap()), //
            make_stage<smoothing_function_1>(                  //
                arg_phi(),                                     //
                arg_laplap(),                                  //
                arg_phi_new())                                 //
            ));                                                //
}
