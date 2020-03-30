#include <gridtools/storage/builder.hpp>
#include <iostream>

using namespace gridtools;

#ifdef GT_CUDACC
#include <gridtools/storage/cuda.hpp>
using traits_t = storage::cuda;
#else
#include <gridtools/storage/mc.hpp>
using traits_t = storage::mc;
#endif

int main() {
    uint_t Ni = 10;
    uint_t Nj = 12;
    uint_t Nk = 20;

    auto const builder = storage::builder<traits_t>.dimensions(Ni, Nj, Nk).id<0>();
    auto phi = builder
                   .name("phi")                                                //
                   .type<float>()                                              //
                   .initializer([](int i, int j, int k) { return i + j + k; }) //
                   .build();
    auto lap = builder.name("lap").type<double>().value(-1).build();

    std::cout << phi->name() << "\n";

    auto phi_view = phi->host_view();
    phi_view(1, 2, 3) = 3.1415;
    std::cout << "phi_view(1, 2, 3) = " << phi_view(1, 2, 3) << std::endl;
    std::cout << "j length = " << phi->lengths()[1] << std::endl;
}
