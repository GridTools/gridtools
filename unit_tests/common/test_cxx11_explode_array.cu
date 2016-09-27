#include "gtest/gtest.h"

#include "test_cxx11_explode_array.hpp"

__global__
void explode_static_kernel(bool* result)
{
    *result = test_explode_static();
}

__global__
void explode_with_object_kernel(bool* result)
{
    *result = test_explode_with_object();
}

__global__
void explode_with_tuple_kernel(bool* result)
{
    *result = test_explode_with_tuple();
}

__global__
void explode_with_tuple_with_object_kernel(bool* result)
{
    *result = test_explode_with_tuple_with_object();
}

TEST(explode_array, test_explode_static) {
    bool result;
    bool* resultDevice;
    cudaMalloc(&resultDevice, sizeof(bool));

    // clang-format off
    explode_static_kernel<<<1,1>>>(resultDevice);
    // clang-format on

    cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result);
}

TEST(explode_array, test_explode_with_object) {
    bool result;
    bool* resultDevice;
    cudaMalloc(&resultDevice, sizeof(bool));

    // clang-format off
    explode_with_object_kernel<<<1,1>>>(resultDevice);
    // clang-format on

    cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result);
}

TEST(explode_array, test_explode_with_tuple) {
    bool result;
    bool* resultDevice;
    cudaMalloc(&resultDevice, sizeof(bool));

    // clang-format off
    explode_with_tuple_kernel<<<1,1>>>(resultDevice);
    // clang-format on

    cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result);
}

TEST(explode_array, test_explode_with_tuple_with_object) {
    bool result;
    bool* resultDevice;
    cudaMalloc(&resultDevice, sizeof(bool));

    // clang-format off
    explode_with_tuple_with_object_kernel<<<1,1>>>(resultDevice);
    // clang-format on

    cudaMemcpy(&result, resultDevice, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result);
}
