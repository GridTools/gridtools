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
#include <gridtools/storage/common/storage_info.hpp>
#include <gridtools/storage/data_store.hpp>
#include <gridtools/storage/data_view.hpp>
#include <gridtools/storage/storage_host/host_storage.hpp>

#include "../../tools/multiplet.hpp"

using namespace gridtools;

TEST(DataViewTest, Simple) {
    typedef storage_info<0, layout_map<2, 1, 0>> storage_info_t;
    typedef data_store<host_storage<double>, storage_info_t> data_store_t;
    // create and allocate a data_store
    storage_info_t si(3, 5, 7);
    data_store_t ds(si);
    // create a rw view and fill with some data
    auto dv = make_host_view(ds);
    dv(0, 0, 0) = 50;
    dv(0, 0, 1) = 60;

    // check if interface works
    EXPECT_TRUE(si.lengths() == dv.lengths());

    // check if data is there
    EXPECT_EQ(50, dv(0, 0, 0));
    EXPECT_EQ(dv(0, 0, 1), 60);
    // check if the user protections are working
    EXPECT_EQ(si.index(1, 0, 0), 1);

    EXPECT_EQ(si.index(1, 0, 1), 16);
    // create a ro view
    auto dvro = make_host_view<access_mode::read_only>(ds);
    // check if data is the same
    EXPECT_EQ(50, dvro(0, 0, 0));
    EXPECT_EQ(dvro(0, 0, 1), 60);
    // views are valid (ds <--> dv and ds <--> dvro)
    EXPECT_TRUE(check_consistency(ds, dv));
    EXPECT_TRUE(check_consistency(ds, dvro));

    // create  a second storage
    data_store_t ds_tmp(si);
    // again create a view
    auto dv_tmp = make_host_view<access_mode::read_write>(ds_tmp);
    // the combination ds_tmp <--> dv/dvro is not a valid view
    EXPECT_FALSE(check_consistency(ds, dv_tmp));
    EXPECT_FALSE(check_consistency(ds_tmp, dv));
    EXPECT_FALSE(check_consistency(ds_tmp, dvro));
    EXPECT_TRUE(check_consistency(ds_tmp, dv_tmp));
}

TEST(DataViewTest, ArrayAPI) {
    typedef storage_info<0, layout_map<0, 1, 2>> storage_info_t;
    storage_info_t si(2, 2, 2);

    typedef data_store<host_storage<double>, storage_info_t> data_store_t;
    // create and allocate a data_store
    data_store_t ds(si);
    auto dvro = make_host_view<access_mode::read_write>(ds);

    dvro({1, 1, 1}) = 2.0;
    EXPECT_TRUE((dvro(array<int, 3>{(int)1, (int)1, (int)1}) == 2.0));
}

TEST(DataViewTest, Looping) {
    typedef storage_info<0, layout_map<0, 1, 2>, halo<1, 2, 3>> storage_info_t;
    storage_info_t si(2 + 2, 2 + 4, 2 + 6);

    typedef data_store<host_storage<triplet>, storage_info_t> data_store_t;

    data_store_t ds(si, [](int i, int j, int k) { return triplet{i, j, k}; }, "ds");
    auto view = make_host_view<access_mode::read_write>(ds);

    auto &&lengths = view.lengths();
    for (int i = 0; i < lengths[0]; ++i)
        for (int j = 0; j < lengths[1]; ++j)
            for (int k = 0; k < lengths[2]; ++k)
                EXPECT_EQ(view(i, j, k), (triplet{i, j, k}));
}
