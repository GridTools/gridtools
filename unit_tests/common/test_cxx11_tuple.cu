#include "gtest/gtest.h"
#include "test_cxx11_tuple.hpp"

#ifdef CXX11_ENABLED

__global__
void test_tuple_kernel(bool* result)
{
    test_tuple_elements(result);
}

TEST(tuple, test_elements) {
    bool result;
    bool* resultDevice;
    cudaMalloc(&resultDevice, sizeof(bool));

    // clang-format off
    test_tuple_kernel<<<1,1>>>(resultDevice);
    // clang-format on

    cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result);
}

#endif
