#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/cuda.hpp>

int main() {
    auto builder = gridtools::storage::builder<gridtools::storage::cuda>.type<int>().dimensions(2);
    auto my_storage = builder();
    my_storage->host_view()(0) = 10;
    my_storage->get_target_ptr();
}
