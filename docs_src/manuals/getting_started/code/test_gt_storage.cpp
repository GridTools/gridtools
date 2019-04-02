#include <gridtools/storage/storage_facility.hpp>
#include <iostream>

using namespace gridtools;

#ifdef __CUDACC__
using backend_t = backend::cuda;
#else
using backend_t = backend::mc;
#endif

using storage_info_t = storage_traits<backend_t>::storage_info_t<0, 3>;
using data_store_t = storage_traits<backend_t>::data_store_t<double, storage_info_t>;

int main() {
    uint_t Ni = 10;
    uint_t Nj = 12;
    uint_t Nk = 20;

    storage_info_t info(Ni, Nj, Nk);

    data_store_t phi(info, -1., "phi");
    data_store_t lap(info, -1., "lap");

    std::cout << phi.name() << "\n";

    auto phi_view = make_host_view(phi);
    phi_view(1, 2, 3) = 3.1415;
    std::cout << "phi_view(1, 2, 3) = " << phi_view(1, 2, 3) << std::endl;

    phi.sync();
}
