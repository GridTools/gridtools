/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../tools/multiplet.hpp"
#include "gtest/gtest.h"
#include <gridtools/common/gt_assert.hpp>
#include <gridtools/storage/data_store.hpp>
#include <gridtools/storage/data_view.hpp>
#include <gridtools/storage/storage_cuda/cuda_storage.hpp>

using namespace gridtools;

const int c_x = 3 /* < 32 for this test */, c_y = 5, c_z = 7;

template <typename View>
__global__ void mul2(View s) {
    using storage_info_t = typename View::storage_info_t;
    bool correct_dims = (s.template total_length<0>() == c_x) && (s.template total_length<1>() == c_y) &&
                        (s.template total_length<2>() == c_z);
    bool correct_size = (s.padded_total_length() == storage_info_t::alignment_t::value * c_y * c_z);
    s(0, 0, 0) *= (2 * correct_dims * correct_size);
    s(1, 0, 0) *= (2 * correct_dims * correct_size);
}

TEST(DataViewTest, Simple) {
    typedef storage_info<0, layout_map<2, 1, 0>, halo<0, 0, 0>, alignment<32>> storage_info_t;
    typedef data_store<cuda_storage<double>, storage_info_t> data_store_t;
    // create and allocate a data_store
    constexpr storage_info_t si(c_x, c_y, c_z);
    data_store_t ds(si);
    // create a rw view and fill with some data
    auto dv = make_host_view(ds);
    dv(0, 0, 0) = 50;
    dv(1, 0, 0) = 60;

    // check if interface works
    ASSERT_TRUE((si.length<0>() == dv.length<0>()));
    ASSERT_TRUE((si.length<1>() == dv.length<1>()));
    ASSERT_TRUE((si.length<2>() == dv.length<2>()));

    ASSERT_TRUE((si.total_length<0>() == dv.total_length<0>()));
    ASSERT_TRUE((si.total_length<1>() == dv.total_length<1>()));
    ASSERT_TRUE((si.total_length<2>() == dv.total_length<2>()));

    ASSERT_TRUE((si.begin<0>() == dv.begin<0>()));
    ASSERT_TRUE((si.begin<1>() == dv.begin<1>()));
    ASSERT_TRUE((si.begin<2>() == dv.begin<2>()));

    ASSERT_TRUE((si.total_begin<0>() == dv.total_begin<0>()));
    ASSERT_TRUE((si.total_begin<1>() == dv.total_begin<1>()));
    ASSERT_TRUE((si.total_begin<2>() == dv.total_begin<2>()));

    ASSERT_TRUE((si.end<0>() == dv.end<0>()));
    ASSERT_TRUE((si.end<1>() == dv.end<1>()));
    ASSERT_TRUE((si.end<2>() == dv.end<2>()));

    ASSERT_TRUE((si.total_end<0>() == dv.total_end<0>()));
    ASSERT_TRUE((si.total_end<1>() == dv.total_end<1>()));
    ASSERT_TRUE((si.total_end<2>() == dv.total_end<2>()));

    ASSERT_TRUE((si.padded_total_length() == dv.padded_total_length()));

    ASSERT_TRUE(si.index(1, 0, 1) == c_y * storage_info_t::alignment_t::value + 1);
    // check if data is there
    EXPECT_EQ(50, dv(0, 0, 0));
    EXPECT_EQ(dv(1, 0, 0), 60);
    // create a ro view
    auto dvro = make_host_view<access_mode::read_only>(ds);
    // check if data is the same
    EXPECT_EQ(50, dvro(0, 0, 0));
    EXPECT_EQ(dvro(1, 0, 0), 60);
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
    EXPECT_EQ(dvro(1, 0, 0), 120);

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

    for (int i = view.begin<0>(); i <= view.end<0>(); ++i) {
        for (int j = view.begin<1>(); j <= view.end<1>(); ++j) {
            for (int k = view.begin<2>(); k <= view.end<2>(); ++k) {
                EXPECT_EQ(view(i, j, k), (triplet{i, j, k}));
            }
        }
    }

    for (int i = view.total_begin<0>(); i <= view.total_end<0>(); ++i) {
        for (int j = view.total_begin<1>(); j <= view.total_end<1>(); ++j) {
            for (int k = view.total_begin<2>(); k <= view.total_end<2>(); ++k) {
                EXPECT_EQ(view(i, j, k), (triplet{i, j, k}));
            }
        }
    }
}
