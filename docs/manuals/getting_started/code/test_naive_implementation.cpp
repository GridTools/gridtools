#include <gridtools/storage/storage-facility.hpp>

using namespace gridtools;

using target_t = target::mc;

using storage_info_t = storage_traits<target_t>::storage_info_t<0, 3>;
using data_store_t = storage_traits<target_t>::data_store_t<double, storage_info_t>;

using storage_view_t = decltype(make_host_view(std::declval<data_store_t>()));

// lap-begin
void laplacian(storage_view_t &lap, storage_view_t &in, int boundary_size) {
    int Ni = in.total_length<0>();
    int Nj = in.total_length<1>();
    int Nk = in.total_length<2>();
    for (int i = boundary_size; i < Ni - boundary_size; ++i) {
        for (int j = boundary_size; j < Nj - boundary_size; ++j) {
            for (int k = boundary_size; k < Nk - boundary_size; ++k) {
                lap(i, j, k) = -4.0 * in(i, j, k)                  //
                               + in(i + 1, j, k) + in(i - 1, j, k) //
                               + in(i, j + 1, k) + in(i, j - 1, k);
            }
        }
    }
}
// lap-end

// smoothing-begin
void naive_smoothing(storage_view_t &out, storage_view_t &in, double alpha, int kmax) {
    int lap_boundary = 1;
    int full_boundary = 2;

    int Ni = in.total_length<0>();
    int Nj = in.total_length<1>();
    int Nk = in.total_length<2>();

    // Instantiate temporary fields
    storage_info_t info(Ni, Nj, Nk);
    data_store_t lap_storage(info);
    auto lap = make_host_view(lap_storage);
    data_store_t laplap_storage(info);
    auto laplap = make_host_view(laplap_storage);

    // laplacian of phi
    laplacian(lap, in, lap_boundary);
    // laplacian of lap
    laplacian(laplap, lap, full_boundary);

    for (int i = full_boundary; i < Ni - full_boundary; ++i) {
        for (int j = full_boundary; j < Nj - full_boundary; ++j) {
            for (int k = full_boundary; k < Nk - full_boundary; ++k) {
                if (k < kmax)
                    out(i, j, k) = in(i, j, k) - alpha * laplap(i, j, k);
                else
                    out(i, j, k) = in(i, j, k);
            }
        }
    }
}
// smoothing-end

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
