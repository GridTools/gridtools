/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#include <gtest/gtest.h>

#include "test_cuda_is_ptr.cpp"

TEST(test_is_gpu_ptr, cuda_ptr_is_cuda_ptr) {
    double *ptr;
    cudaError_t error = cudaMalloc(&ptr, sizeof(double));
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(-1);
    }

    ASSERT_TRUE(gridtools::is_gpu_ptr(ptr));

    cudaFree(ptr);
}

TEST(test_is_gpu_ptr, cudaMallocHost_ptr_is_not_cuda_ptr) {
    double *ptr;
    cudaError_t error = cudaMallocHost(&ptr, sizeof(double));
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(-1);
    }

    ASSERT_FALSE(gridtools::is_gpu_ptr(ptr));

    cudaFreeHost(ptr);
}

TEST(test_is_gpu_ptr, cuda_ptr_inner_region_are_cuda_ptr) {
    double *ptr;
    const size_t n_elem = 32;
    cudaError_t error = cudaMalloc(&ptr, sizeof(double) * n_elem);
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(-1);
    }

    for (size_t i = 0; i < n_elem; ++i) {
        ASSERT_TRUE(gridtools::is_gpu_ptr(&ptr[i]));
    }
    ASSERT_FALSE(gridtools::is_gpu_ptr(&ptr[n_elem + 1])); // out of bounds

    cudaFree(ptr);
}
