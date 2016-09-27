#include "gtest/gtest.h"
#include "test_cxx11_accumulate.hpp"

__global__ void accumulate_and_kernel(bool *result) { *result = test_accumulate_and(); }

__global__ void accumulate_or_kernel(bool *result) { *result = test_accumulate_or(); }

TEST(accumulate, test_and) {
    bool result;
    bool *resultDevice;
    cudaMalloc(&resultDevice, sizeof(bool));

    // clang-format off
    accumulate_and_kernel<<<1,1>>>(resultDevice);
    // clang-format on

    cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result);
}

TEST(accumulate, test_or) {
    bool result;
    bool *resultDevice;
    cudaMalloc(&resultDevice, sizeof(bool));

    // clang-format off
    accumulate_or_kernel<<<1,1>>>(resultDevice);
    // clang-format on

    cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result);
}
