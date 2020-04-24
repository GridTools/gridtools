#include <gridtools/common/defs.hpp>
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

using namespace gridtools;
using namespace cartesian;

#ifdef GT_CUDACC
#include <gridtools/stencil/backend/cuda.hpp>
#include <gridtools/storage/cuda.hpp>
using backend_t = cuda::backend<>;
using storage_traits_t = storage::cuda;
#else
#include <gridtools/stencil/backend/mc.hpp>
#include <gridtools/storage/mc.hpp>
using backend_t = mc::backend<>;
using storage_traits_t = storage::mc;
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

    run_single_stage(lap_function(), backend_t(), grid, phi, lap);
}
