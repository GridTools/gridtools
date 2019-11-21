#include <gridtools/storage/builder.hpp>
#include <iostream>

using namespace gridtools;

#ifdef __CUDACC__
#include <gridtools/storage/cuda.hpp>
using storage_traits_t = storage::cuda;
#else
#include <gridtools/storage/mc.hpp>
using storage_traits_t = storage::mc;
#endif

int main() {
    uint_t Ni = 10;
    uint_t Nj = 12;
    uint_t Nk = 20;

    const auto builder = storage::builder<storage_traits_t>.type<double>().dimensions(Ni, Nj, Nk);

    auto phi = builder.name("phi").value(-1).build();
    auto lap = builder.name("lap").value(-1).build();

    std::cout << phi->name() << "\n";

    auto phi_view = phi->host_view();
    phi_view(1, 2, 3) = 3.1415;
    std::cout << "phi_view(1, 2, 3) = " << phi_view(1, 2, 3) << std::endl;
} // end
