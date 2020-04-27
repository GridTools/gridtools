#include <gridtools/common/defs.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

using namespace gridtools;
using namespace stencil;
using namespace cartesian;

#ifdef GT_CUDACC
#include <gridtools/stencil/gpu.hpp>
#include <gridtools/storage/gpu.hpp>
using stencil_backend_t = stencil::gpu<>;
using storage_traits_t = storage::gpu;
#else
#include <gridtools/stencil/cpu_ifirst.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>
using stencil_backend_t = stencil::cpu_ifirst<>;
using storage_traits_t = storage::cpu_ifirst;
#endif

constexpr dimension<1> i;
constexpr dimension<2> j;

struct lap_function {
    using in = in_accessor<0, extent<-1, 1, -1, 1>>;
    using lap = inout_accessor<1>;

    using param_list = make_param_list<in, lap>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &&eval) {
        eval(lap(i, j)) = -4 * eval(in(i, j))   //
                          + eval(in(i + 1, j))  //
                          + eval(in(i, j + 1))  //
                          + eval(in(i - 1, j))  //
                          + eval(in(i, j - 1)); //
    }
};

int main() {
    uint_t Ni = 10;
    uint_t Nj = 12;
    uint_t Nk = 20;
    int halo = 1;

    auto builder = storage::builder<storage_traits_t> //
                       .type<double>()                //
                       .dimensions(Ni, Nj, Nk)        //
                       .halos(halo, halo, 0);         //

    auto phi = builder();
    auto lap = builder();

    halo_descriptor boundary_i(halo, halo, halo, Ni - halo - 1, Ni);
    halo_descriptor boundary_j(halo, halo, halo, Nj - halo - 1, Nj);
    auto grid = make_grid(boundary_i, boundary_j, Nk);

    run_single_stage(lap_function(), stencil_backend_t(), grid, phi, lap);
}
