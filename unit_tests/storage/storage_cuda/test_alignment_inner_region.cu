/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/storage_info_extender.hpp>
#include <gridtools/storage/storage-facility.hpp>
#include <gtest/gtest.h>
#include <iostream>

namespace gt = gridtools;

template <typename View, typename Ptr>
__global__ void check(View view, Ptr *pgres, gt::uint_t h1, gt::uint_t h2, gt::uint_t h3, gt::uint_t a) {
    *pgres = &view(h1, h2, h3);
}

template <typename Layout, gt::int_t I>
constexpr gt::uint_t add_or_not(gt::uint_t x) {
    return (Layout::find(1) == I) ? x : 0;
}

template <typename ValueType, gt::uint_t a, typename Layout>
void run() {
    ValueType **pgres;
    cudaMalloc(&pgres, sizeof(int));

    constexpr gt::uint_t h1 = 7;
    constexpr gt::uint_t h2 = 4;
    constexpr gt::uint_t h3 = 5;
    using info = gt::storage_info_interface<0, Layout, gt::halo<h1, h2, h3>, gt::alignment<a>>;
    using store = gt::storage_traits<gridtools::target::cuda>::data_store_t<ValueType, info>;

    info i(23, 34, 12);
    store s(i);

    auto view = gt::make_device_view(s);

    ValueType *res;

    cudaMemcpy(pgres, &res, sizeof(int), cudaMemcpyHostToDevice);
    check<<<1, 1>>>(view, pgres, h1, h2, h3, a);
    cudaMemcpy(&res, pgres, sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(res) % a, 0);

    cudaMemcpy(pgres, &res, sizeof(int), cudaMemcpyHostToDevice);
    check<<<1, 1>>>(
        view, pgres, h1 + add_or_not<Layout, 0>(1), h2 + add_or_not<Layout, 1>(1), h3 + add_or_not<Layout, 2>(1), a);
    cudaMemcpy(&res, pgres, sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(res) % a, 0);

    cudaMemcpy(pgres, &res, sizeof(int), cudaMemcpyHostToDevice);
    check<<<1, 1>>>(
        view, pgres, h1 + add_or_not<Layout, 0>(2), h2 + add_or_not<Layout, 1>(2), h3 + add_or_not<Layout, 2>(2), a);
    cudaMemcpy(&res, pgres, sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(res) % a, 0);

    cudaFree(pgres);
}

TEST(Storage, InnerRegionAlignmentCharCuda210) { run<char, 1024, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, InnerRegionAlignmentIntCuda210) { run<int, 256, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, InnerRegionAlignmentFloatCuda210) { run<float, 32, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, InnerRegionAlignmentDoubleCuda210) { run<double, 512, gt::layout_map<2, 1, 0>>(); }

TEST(Storage, InnerRegionAlignmentCharCuda012) { run<char, 1024, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, InnerRegionAlignmentIntCuda012) { run<int, 256, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, InnerRegionAlignmentFloatCuda012) { run<float, 32, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, InnerRegionAlignmentDoubleCuda012) { run<double, 512, gt::layout_map<0, 1, 2>>(); }

TEST(Storage, InnerRegionAlignmentCharCuda021) { run<char, 1024, gt::layout_map<0, 2, 1>>(); }

TEST(Storage, InnerRegionAlignmentIntCuda021) { run<int, 256, gt::layout_map<0, 2, 1>>(); }

TEST(Storage, InnerRegionAlignmentFloatCuda021) { run<float, 32, gt::layout_map<0, 2, 1>>(); }

TEST(Storage, InnerRegionAlignmentDoubleCuda021) { run<double, 512, gt::layout_map<0, 2, 1>>(); }
