#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

using namespace gridtools;
using namespace cartesian;

#if defined(__CUDACC__) || defined(__HIPCC__)
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

constexpr dimension<1> i;
constexpr dimension<2> j;

struct lap_function {
    using in = in_accessor<0, extent<-1, 1, -1, 1>>;
    using lap = inout_accessor<1>;

    using param_list = make_param_list<in, lap>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &&eval) {
        eval(lap(i, j)) =
            -4 * eval(in(i, j)) + eval(in(i + 1, j)) + eval(in(i, j + 1)) + eval(in(i - 1, j)) + eval(in(i, j - 1));
    }
};

int main() {
    uint_t Ni = 10;
    uint_t Nj = 12;
    uint_t Nk = 20;

    auto builder = storage::builder<storage_traits_t>.type<double>().dimensions(Ni, Nj, Nk).halos(1, 1, 0).value(-1);

    auto phi = builder.name("phi")();
    auto lap = builder.name("lap")();

    int halo_size = 1;
    halo_descriptor boundary_i(halo_size, halo_size, halo_size, Ni - halo_size - 1, Ni);
    halo_descriptor boundary_j(halo_size, halo_size, halo_size, Nj - halo_size - 1, Nj);
    auto my_grid = make_grid(boundary_i, boundary_j, Nk);

    easy_run(lap_function(), backend_t(), my_grid, phi, lap);
}
