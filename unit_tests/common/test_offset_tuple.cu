#include "gtest/gtest.h"
#include "test_offset_tuple.hpp"

__global__
void test_offset_kernel(bool* result)
{
    test_offset_tuple(result);
}

TEST(offset_tuple, test_offset_tuple) {
    bool result;
    bool* resultDevice;
    cudaMalloc(&resultDevice, sizeof(bool));

    // clang-format off
    test_offset_kernel<<<1,1>>>(resultDevice);
    // clang-format on

    cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result);
}
