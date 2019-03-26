/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../../tools/triplet.hpp"
#include "gtest/gtest.h"
#include <gridtools/common/gt_assert.hpp>
#include <gridtools/storage/data_store.hpp>
#include <gridtools/storage/storage_cuda/cuda_storage.hpp>
#include <gridtools/storage/storage_cuda/cuda_storage_info.hpp>
#include <gridtools/storage/storage_cuda/data_view_helpers.hpp>

using namespace gridtools;

const int c_x = 3 /* < 32 for this test */, c_y = 5, c_z = 7;

template <typename View>
__global__ void mul2(View s) {
    bool correct_dims = (s.template total_length<0>() == c_x) && (s.template total_length<1>() == c_y) &&
                        (s.template total_length<2>() == c_z);
    bool correct_size = (s.padded_total_length() == 32 * c_y * c_z);
    s(0, 0, 0) *= (2 * correct_dims * correct_size);
    s(1, 0, 0) *= (2 * correct_dims * correct_size);
}

TEST(DataViewTest, Simple) {
    typedef cuda_storage_info<0, layout_map<2, 1, 0>> storage_info_t;
    typedef data_store<cuda_storage<double>, storage_info_t> data_store_t;
    // create and allocate a data_store
    constexpr storage_info_t si(c_x, c_y, c_z);
    data_store_t ds(si);
    // create a rw view and fill with some data
    data_view<data_store_t> dv = make_host_view(ds);
    GT_STATIC_ASSERT((is_data_view<decltype(dv)>::value), "is_data_view check failed");
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

    // check if the user protections are working
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ <= 9
    // CUDA10 does not like to evaluate storage_info functions constexpr as a member (m_gpu_ptr) is mutable
    static_assert(si.index(1, 0, 0) == 1, "constexpr index method call failed");
#endif

    ASSERT_TRUE(si.index(1, 0, 1) == c_y * 32 + 1);
    // check if data is there
    EXPECT_EQ(50, dv(0, 0, 0));
    EXPECT_EQ(dv(1, 0, 0), 60);
    // create a ro view
    data_view<data_store_t, access_mode::read_only> dvro = make_host_view<access_mode::read_only>(ds);
    // check if data is the same
    EXPECT_EQ(50, dvro(0, 0, 0));
    EXPECT_EQ(dvro(1, 0, 0), 60);
    // views are valid (ds <--> dv and ds <--> dvro)
    EXPECT_TRUE(check_consistency(ds, dv));
    EXPECT_TRUE(check_consistency(ds, dvro));
    EXPECT_TRUE(dv.valid());
    EXPECT_TRUE(dvro.valid());

    // sync, create a device view and call kernel
    ds.sync();
    auto devv = make_device_view(ds);
    GT_STATIC_ASSERT((is_data_view<decltype(devv)>::value), "is_data_view check failed");
    EXPECT_TRUE(check_consistency(ds, devv));
    EXPECT_FALSE(check_consistency(ds, dv));
    EXPECT_FALSE(check_consistency(ds, dvro));
    EXPECT_TRUE(devv.valid());
    EXPECT_FALSE(dv.valid());
    EXPECT_FALSE(dvro.valid());
    mul2<<<1, 1>>>(devv);

    // sync and check if read only host view is valid
    ds.sync();
    EXPECT_FALSE(check_consistency(ds, devv));
    EXPECT_FALSE(check_consistency(ds, dv));
    EXPECT_TRUE(check_consistency(ds, dvro));
    EXPECT_FALSE(devv.valid());
    EXPECT_FALSE(dv.valid());
    EXPECT_TRUE(dvro.valid());
    // check if data is the same
    EXPECT_EQ(100, dvro(0, 0, 0));
    EXPECT_EQ(dvro(1, 0, 0), 120);

    // create and allocate a second storage
    data_store_t ds_tmp(si);
    // again create a view
    data_view<data_store_t> dv_tmp = make_host_view<access_mode::read_write>(ds_tmp);
    // the combination ds_tmp <--> dv/dvro is not a valid view
    EXPECT_FALSE(check_consistency(ds, dv_tmp));
    EXPECT_FALSE(check_consistency(ds_tmp, devv));
    EXPECT_FALSE(check_consistency(ds_tmp, dvro));
    EXPECT_TRUE(check_consistency(ds_tmp, dv_tmp));

    EXPECT_TRUE(dv_tmp.valid());
    EXPECT_FALSE(devv.valid());
    EXPECT_TRUE(dvro.valid());
    EXPECT_TRUE(dv_tmp.valid());

    // destroy a storage, this should also invalidate the views
    ds.reset();
    EXPECT_FALSE(check_consistency(ds, dv));
    EXPECT_FALSE(check_consistency(ds, dvro));
}

TEST(DataViewTest, ZeroSize) {
    typedef cuda_storage_info<0, layout_map<0>> storage_info_t;
    typedef data_store<cuda_storage<double>, storage_info_t> data_store_t;
    // create and allocate a data_store
    data_store_t ds;
    make_host_view<access_mode::read_only>(ds);
    make_device_view<access_mode::read_only>(ds);
}

TEST(DataViewTest, Looping) {
    typedef cuda_storage_info<0, layout_map<0, 1, 2>, halo<1, 2, 3>> storage_info_t;
    storage_info_t si(2 + 2, 2 + 4, 2 + 6);

    typedef data_store<cuda_storage<triplet>, storage_info_t> data_store_t;

    data_store_t ds(si, [](int i, int j, int k) { return triplet(i, j, k); }, "ds");
    auto view = make_host_view<access_mode::read_write>(ds);

    for (int i = view.begin<0>(); i <= view.end<0>(); ++i) {
        for (int j = view.begin<1>(); j <= view.end<1>(); ++j) {
            for (int k = view.begin<2>(); k <= view.end<2>(); ++k) {
                EXPECT_EQ(view(i, j, k), triplet(i, j, k));
            }
        }
    }

    for (int i = view.total_begin<0>(); i <= view.total_end<0>(); ++i) {
        for (int j = view.total_begin<1>(); j <= view.total_end<1>(); ++j) {
            for (int k = view.total_begin<2>(); k <= view.total_end<2>(); ++k) {
                EXPECT_EQ(view(i, j, k), triplet(i, j, k));
            }
        }
    }
}

TEST(DataViewTest, TargetView) {
    typedef cuda_storage_info<0, layout_map<0, 1, 2>, halo<1, 2, 3>> storage_info_t;
    storage_info_t si(2 + 2, 2 + 4, 2 + 6);

    typedef data_store<cuda_storage<triplet>, storage_info_t> data_store_t;

    data_store_t ds(si, [](int i, int j, int k) { return triplet(i, j, k); }, "ds");

    auto target_view = make_target_view<access_mode::read_only>(ds);
    auto device_view = make_device_view<access_mode::read_only>(ds);

    ASSERT_EQ(advanced::get_raw_pointer_of(device_view), advanced::get_raw_pointer_of(target_view));
}

TEST(DataViewTest, CheckMemorySpace) {
    typedef cuda_storage_info<0, layout_map<0, 1, 2>, halo<1, 2, 3>> storage_info_t;
    storage_info_t si(2 + 2 * 1, 2 + 2 * 3, 2 + 2 * 3);

    typedef data_store<cuda_storage<int>, storage_info_t> data_store_t;

    data_store_t ds(si, -1, "ds");
    auto view = make_device_view<access_mode::read_write>(ds);

    EXPECT_THROW(view(0, 0, 1), std::runtime_error);
}
