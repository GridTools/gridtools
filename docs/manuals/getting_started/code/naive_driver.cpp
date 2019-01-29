#include <gridtools/storage/storage-facility.hpp>

using namespace gridtools;

using target_t = target::mc;

using storage_info_t = storage_traits<target_t>::storage_info_t<0, 3>;
using data_store_t = storage_traits<target_t>::data_store_t<double, storage_info_t>;

using storage_view_t = decltype(make_host_view(std::declval<data_store_t>()));

#include "naive_laplacian.hpp"
#include "naive_smoothing.hpp"

int main() {
    uint_t Ni = 10;
    uint_t Nj = 12;
    uint_t Nk = 20;
    uint_t kmax = 12;

    storage_info_t info(Ni, Nj, Nk);

    data_store_t phi(info);
    data_store_t phi_new(info);

    auto phi_view = make_host_view(phi);
    auto phi_new_view = make_host_view(phi_new);

    laplacian(phi_new_view, phi_view, 1);
    naive_smoothing(phi_new_view, phi_view, 0.5, kmax);
}
