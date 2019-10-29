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

#include <gridtools/common/gt_assert.hpp>
#include <gridtools/storage/data_store.hpp>
#include <gridtools/storage/data_view.hpp>
#include <gridtools/storage/storage_cuda/cuda_storage.hpp>

#include "../../tools/multiplet.hpp"

using namespace gridtools;

constexpr int c_x = 3 /* < 32 for this test */, c_y = 5, c_z = 7;
using storage_info_t = storage_info<0, layout_map<2, 1, 0>, halo<0, 0, 0>, alignment<32>>;

template <class View>
__global__ void mul2(View s) {
    auto &&lengths = s.lengths();
    bool expected_dims = lengths[0] == c_x && lengths[1] == c_y && lengths[2] == c_z;
    bool expected_size = s.length() <= storage_info_t::alignment_t::value * c_y * c_z && s.length() >= c_x * c_y * c_z;
    s(0, 0, 0) *= 2 * expected_dims * expected_size;
    s(1, 0, 0) *= 2 * expected_dims * expected_size;
}

TEST(DataViewTest, Simple) {
    typedef data_store<cuda_storage<double>, storage_info_t> data_store_t;
    // create and allocate a data_store
    storage_info_t si(c_x, c_y, c_z);
    data_store_t ds(si);
    // create a rw view and fill with some data
    auto dv = make_host_view(ds);
    dv(0, 0, 0) = 50;
    dv(1, 0, 0) = 60;

    // check if interface works
    EXPECT_TRUE(si.lengths() == dv.lengths());

    EXPECT_EQ(si.index(1, 0, 1), c_y * storage_info_t::alignment_t::value + 1);
    // check if data is there
    EXPECT_EQ(50, dv(0, 0, 0));
    EXPECT_EQ(60, dv(1, 0, 0));
    // create a ro view
    auto dvro = make_host_view<access_mode::read_only>(ds);
    // check if data is the same
    EXPECT_EQ(50, dvro(0, 0, 0));
    EXPECT_EQ(60, dvro(1, 0, 0));
    // views are valid (ds <--> dv and ds <--> dvro)
    EXPECT_TRUE(check_consistency(ds, dv));
    EXPECT_TRUE(check_consistency(ds, dvro));

    // sync, create a device view and call kernel
    ds.sync();
    auto devv = make_target_view(ds);
    EXPECT_TRUE(check_consistency(ds, devv));
    EXPECT_FALSE(check_consistency(ds, dv));
    EXPECT_FALSE(check_consistency(ds, dvro));
    mul2<<<1, 1>>>(devv);

    // sync and check if read only host view is valid
    ds.sync();
    EXPECT_FALSE(check_consistency(ds, devv));
    EXPECT_FALSE(check_consistency(ds, dv));
    EXPECT_TRUE(check_consistency(ds, dvro));
    // check if data is the same
    EXPECT_EQ(100, dvro(0, 0, 0));
    EXPECT_EQ(120, dvro(1, 0, 0));

    // create and allocate a second storage
    data_store_t ds_tmp(si);
    // again create a view
    auto dv_tmp = make_host_view<access_mode::read_write>(ds_tmp);
    // the combination ds_tmp <--> dv/dvro is not a valid view
    EXPECT_FALSE(check_consistency(ds, dv_tmp));
    EXPECT_FALSE(check_consistency(ds_tmp, devv));
    EXPECT_FALSE(check_consistency(ds_tmp, dvro));
    EXPECT_TRUE(check_consistency(ds_tmp, dv_tmp));
}

TEST(DataViewTest, Looping) {
    typedef storage_info<0, layout_map<0, 1, 2>, halo<1, 2, 3>, alignment<32>> storage_info_t;
    storage_info_t si(2 + 2, 2 + 4, 2 + 6);

    typedef data_store<cuda_storage<triplet>, storage_info_t> data_store_t;

    data_store_t ds(si, [](int i, int j, int k) { return triplet{i, j, k}; }, "ds");
    auto view = make_host_view<access_mode::read_write>(ds);

    auto &&lengths = view.lengths();
    for (int i = 0; i < lengths[0]; ++i)
        for (int j = 0; j < lengths[1]; ++j)
            for (int k = 0; k < lengths[2]; ++k)
                EXPECT_EQ(view(i, j, k), (triplet{i, j, k}));
}
