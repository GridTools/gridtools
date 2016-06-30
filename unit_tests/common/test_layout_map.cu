#include "gtest/gtest.h"
#include "test_layout_map.hpp"

__global__
void test_layout_accessors_kernel(bool* result)
{
    test_layout_accessors(result);
}

__global__
void test_layout_findval_kernel(bool* result)
{
    test_layout_find_val(result);
}

TEST(layout_map_cuda, test_layout_accessors) {
    bool result;
    bool* resultDevice;
    cudaMalloc(&resultDevice, sizeof(bool));

    // clang-format off
    test_layout_accessors_kernel<<<1,1>>>(resultDevice);
    // clang-format on

    cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result);
}

TEST(layout_map_cuda, test_layout_findval) {
    bool result;
    bool* resultDevice;
    cudaMalloc(&resultDevice, sizeof(bool));

    // clang-format off
    test_layout_findval_kernel<<<1,1>>>(resultDevice);
    // clang-format on

    cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result);
}
