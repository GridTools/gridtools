/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include "test_layout_transformation.cpp"

using namespace gridtools;

TEST(layout_transformation, 3D_reverse_layout_cuda) {
    uint_t Nx = 4;
    uint_t Ny = 5;
    uint_t Nz = 6;

    std::vector<uint_t> dims{Nx, Ny, Nz};
    std::vector<uint_t> src_strides{1, Nx, Nx * Ny};
    std::vector<uint_t> dst_strides{Ny * Nz, Nz, 1};

    Index src_index(dims, src_strides);
    double *src = new double[src_index.size()];
    double *d_src;
    GT_CUDA_CHECK(cudaMalloc(&d_src, sizeof(double) * src_index.size()));
    init<3>(src, src_index, [](const array<size_t, 3> &a) { return a[0] * 100 + a[1] * 10 + a[2]; });
    GT_CUDA_CHECK(cudaMemcpy(d_src, src, sizeof(double) * src_index.size(), cudaMemcpyHostToDevice));

    Index dst_index(dims, dst_strides);
    double *dst = new double[dst_index.size()];
    double *d_dst;
    GT_CUDA_CHECK(cudaMalloc(&d_dst, sizeof(double) * dst_index.size()));
    init<3>(dst, dst_index, [](const array<size_t, 3> &) { return -1; });
    GT_CUDA_CHECK(cudaMemcpy(d_dst, dst, sizeof(double) * dst_index.size(), cudaMemcpyHostToDevice));

    gridtools::interface::transform(d_dst, d_src, dims, dst_strides, src_strides);

    GT_CUDA_CHECK(cudaMemcpy(dst, d_dst, sizeof(double) * src_index.size(), cudaMemcpyDeviceToHost));
    verify<3>(src, src_index, dst, dst_index);

    delete[] src;
    delete[] dst;
    GT_CUDA_CHECK(cudaFree(d_src));
    GT_CUDA_CHECK(cudaFree(d_dst));
}

TEST(layout_transformation, 2D_reverse_layout_cuda) {
    uint_t Nx = 4;
    uint_t Ny = 5;

    std::vector<uint_t> dims{Nx, Ny};
    std::vector<uint_t> src_strides{1, Nx};
    std::vector<uint_t> dst_strides{Ny, 1};

    Index src_index(dims, src_strides);
    double *src = new double[src_index.size()];
    double *d_src;
    GT_CUDA_CHECK(cudaMalloc(&d_src, sizeof(double) * src_index.size()));
    init<2>(src, src_index, [](const array<size_t, 2> &a) { return a[0] * 10 + a[1]; });
    GT_CUDA_CHECK(cudaMemcpy(d_src, src, sizeof(double) * src_index.size(), cudaMemcpyHostToDevice));

    Index dst_index(dims, dst_strides);
    double *dst = new double[dst_index.size()];
    double *d_dst;
    GT_CUDA_CHECK(cudaMalloc(&d_dst, sizeof(double) * dst_index.size()));
    init<2>(dst, dst_index, [](const array<size_t, 2> &) { return -1; });
    GT_CUDA_CHECK(cudaMemcpy(d_dst, dst, sizeof(double) * dst_index.size(), cudaMemcpyHostToDevice));

    gridtools::interface::transform(d_dst, d_src, dims, dst_strides, src_strides);

    GT_CUDA_CHECK(cudaMemcpy(dst, d_dst, sizeof(double) * src_index.size(), cudaMemcpyDeviceToHost));
    verify<2>(src, src_index, dst, dst_index);

    delete[] src;
    delete[] dst;
    GT_CUDA_CHECK(cudaFree(d_src));
    GT_CUDA_CHECK(cudaFree(d_dst));
}

TEST(layout_transformation, 4D_reverse_layout_cuda) {
    uint_t Nx = 4;
    uint_t Ny = 5;
    uint_t Nz = 6;
    uint_t Nw = 7;

    std::vector<uint_t> dims{Nx, Ny, Nz, Nw};
    std::vector<uint_t> src_strides{1, Nx, Nx * Ny, Nx * Ny * Nz};
    std::vector<uint_t> dst_strides{Ny * Nz * Nw, Nz * Nw, Nw, 1};

    Index src_index(dims, src_strides);
    double *src = new double[src_index.size()];
    double *d_src;
    GT_CUDA_CHECK(cudaMalloc(&d_src, sizeof(double) * src_index.size()));
    init<4>(src, src_index, [](const array<size_t, 4> &a) { return a[0] * 1000 + a[1] * 100 + a[2] * 10 + a[3]; });
    GT_CUDA_CHECK(cudaMemcpy(d_src, src, sizeof(double) * src_index.size(), cudaMemcpyHostToDevice));

    Index dst_index(dims, dst_strides);
    double *dst = new double[dst_index.size()];
    double *d_dst;
    GT_CUDA_CHECK(cudaMalloc(&d_dst, sizeof(double) * dst_index.size()));
    init<4>(dst, dst_index, [](const array<size_t, 4> &) { return -1.; });
    GT_CUDA_CHECK(cudaMemcpy(d_dst, dst, sizeof(double) * dst_index.size(), cudaMemcpyHostToDevice));

    gridtools::interface::transform(d_dst, d_src, dims, dst_strides, src_strides);

    GT_CUDA_CHECK(cudaMemcpy(dst, d_dst, sizeof(double) * src_index.size(), cudaMemcpyDeviceToHost));
    verify<4>(src, src_index, dst, dst_index);

    delete[] src;
    delete[] dst;
    GT_CUDA_CHECK(cudaFree(d_src));
    GT_CUDA_CHECK(cudaFree(d_dst));
}
