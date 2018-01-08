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

#include "test_layout_transformation.cpp"

using namespace gridtools;

TEST(layout_transformation, 3D_reverse_layout_cuda) {
    uint_t Nx = 4;
    uint_t Ny = 5;
    uint_t Nz = 6;

    std::vector< uint_t > dims{Nx, Ny, Nz};
    std::vector< uint_t > src_strides{1, Nx, Nx * Ny};
    std::vector< uint_t > dst_strides{Ny * Nz, Nz, 1};

    Index src_index(dims, src_strides);
    double *src = new double[src_index.size()];
    double *d_src;
    cudaMalloc(&d_src, sizeof(double) * src_index.size());
    init< 3 >(src, src_index, [](int i, int j, int k) { return i * 100 + j * 10 + k; });
    cudaMemcpy(d_src, src, sizeof(double) * src_index.size(), cudaMemcpyHostToDevice);

    Index dst_index(dims, dst_strides);
    double *dst = new double[dst_index.size()];
    double *d_dst;
    cudaMalloc(&d_dst, sizeof(double) * dst_index.size());
    init< 3 >(dst, dst_index, [](int i, int j, int k) { return -1; });
    cudaMemcpy(d_dst, dst, sizeof(double) * dst_index.size(), cudaMemcpyHostToDevice);

    gridtools::interface::transform(d_dst, d_src, dims, dst_strides, src_strides);

    cudaMemcpy(dst, d_dst, sizeof(double) * src_index.size(), cudaMemcpyDeviceToHost);
    verify< 3 >(src, src_index, dst, dst_index);

    delete[] src;
    delete[] dst;
    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST(layout_transformation, 2D_reverse_layout_cuda) {
    uint_t Nx = 4;
    uint_t Ny = 5;

    std::vector< uint_t > dims{Nx, Ny};
    std::vector< uint_t > src_strides{1, Nx};
    std::vector< uint_t > dst_strides{Ny, 1};

    Index src_index(dims, src_strides);
    double *src = new double[src_index.size()];
    double *d_src;
    cudaMalloc(&d_src, sizeof(double) * src_index.size());
    init< 2 >(src, src_index, [](int i, int j) { return i * 10 + j; });
    cudaMemcpy(d_src, src, sizeof(double) * src_index.size(), cudaMemcpyHostToDevice);

    Index dst_index(dims, dst_strides);
    double *dst = new double[dst_index.size()];
    double *d_dst;
    cudaMalloc(&d_dst, sizeof(double) * dst_index.size());
    init< 2 >(dst, dst_index, [](int i, int j) { return -1; });
    cudaMemcpy(d_dst, dst, sizeof(double) * dst_index.size(), cudaMemcpyHostToDevice);

    gridtools::interface::transform(d_dst, d_src, dims, dst_strides, src_strides);

    cudaMemcpy(dst, d_dst, sizeof(double) * src_index.size(), cudaMemcpyDeviceToHost);
    verify< 2 >(src, src_index, dst, dst_index);

    delete[] src;
    delete[] dst;
    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST(layout_transformation, 4D_reverse_layout_cuda) {
    uint_t Nx = 4;
    uint_t Ny = 5;
    uint_t Nz = 6;
    uint_t Nw = 7;

    std::vector< uint_t > dims{Nx, Ny, Nz, Nw};
    std::vector< uint_t > src_strides{1, Nx, Nx * Ny, Nx * Ny * Nz};
    std::vector< uint_t > dst_strides{Ny * Nz * Nw, Nz * Nw, Nw, 1};

    Index src_index(dims, src_strides);
    double *src = new double[src_index.size()];
    double *d_src;
    cudaMalloc(&d_src, sizeof(double) * src_index.size());
    init< 4 >(src, src_index, [](int i, int j, int k, int l) { return i * 1000 + j * 100 + k * 10 + l; });
    cudaMemcpy(d_src, src, sizeof(double) * src_index.size(), cudaMemcpyHostToDevice);

    Index dst_index(dims, dst_strides);
    double *dst = new double[dst_index.size()];
    double *d_dst;
    cudaMalloc(&d_dst, sizeof(double) * dst_index.size());
    init< 4 >(dst, dst_index, [](int i, int j, int k, int l) { return -1; });
    cudaMemcpy(d_dst, dst, sizeof(double) * dst_index.size(), cudaMemcpyHostToDevice);

    gridtools::interface::transform(d_dst, d_src, dims, dst_strides, src_strides);

    cudaMemcpy(dst, d_dst, sizeof(double) * src_index.size(), cudaMemcpyDeviceToHost);
    verify< 4 >(src, src_index, dst, dst_index);

    delete[] src;
    delete[] dst;
    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST(layout_transformation, mixing_host_and_device_ptr) {
    double *d_src;
    cudaMalloc(&d_src, sizeof(double));

    double *dst = new double;

    // mixing host pointer + device pointer
    ASSERT_ANY_THROW(gridtools::interface::transform(
        dst, d_src, std::vector< uint_t >(), std::vector< uint_t >(), std::vector< uint_t >()));

    delete[] dst;
    cudaFree(d_src);
}
