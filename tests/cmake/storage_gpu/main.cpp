#include <cuda_runtime.h>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/gpu.hpp>

int main() {
    int result_value = 42;

    auto builder = gridtools::storage::builder<gridtools::storage::gpu>.type<int>().dimensions(2);
    auto my_storage = builder();

    my_storage->host_view()(0) = result_value;
    auto dev_ptr = my_storage->get_target_ptr();

    int host_buffer;
    cudaMemcpy(&host_buffer, dev_ptr, sizeof(int), cudaMemcpyDeviceToHost);

    if (host_buffer == result_value)
        exit(0);
    else
        exit(1);
}
