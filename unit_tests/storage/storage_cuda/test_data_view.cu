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

#include "gtest/gtest.h"

#include <common/gt_assert.hpp>
#include <storage/data_store.hpp>
#include <storage/storage_cuda/data_view_helpers.hpp>
#include <storage/storage_cuda/storage.hpp>
#include <storage/storage_cuda/storage_info.hpp>

using namespace gridtools;

template < typename View >
__global__ void mul2(View s) {
    bool correct_dims = (s.template dim< 0 >() == 32) && (s.template dim< 1 >() == 3) && (s.template dim< 2 >() == 3);
    bool correct_size = (s.size() == 32 * 3 * 3);
    s(0, 0, 0) *= (2 * correct_dims * correct_size);
    s(1, 0, 0) *= (2 * correct_dims * correct_size);
}

TEST(DataViewTest, Simple) {
    typedef cuda_storage_info< 0, layout_map< 2, 1, 0 > > storage_info_t;
    typedef data_store< cuda_storage< double >, storage_info_t > data_store_t;
    // create and allocate a data_store
    storage_info_t si(3, 3, 3);
    data_store_t ds(si);
    // create a rw view and fill with some data
    data_view< data_store_t > dv = make_host_view(ds);
    GRIDTOOLS_STATIC_ASSERT((is_data_view< decltype(dv) >::value), "is_data_view check failed");
    dv(0, 0, 0) = 50;
    dv(1, 0, 0) = 60;

    // check if dim interface works
    ASSERT_TRUE((si.dim< 0 >() == dv.dim< 0 >()));
    ASSERT_TRUE((si.dim< 1 >() == dv.dim< 1 >()));
    ASSERT_TRUE((si.dim< 2 >() == dv.dim< 2 >()));
    ASSERT_TRUE((si.size() == dv.size()));

    // check if the user protections are working
    ASSERT_TRUE(si.index(1, 0, 0) == 1);
    ASSERT_TRUE(si.index(1, 0, 1) == 97);
    // check if data is there
    EXPECT_EQ(50, dv(0, 0, 0));
    EXPECT_EQ(dv(1, 0, 0), 60);
    // create a ro view
    data_view< data_store_t, access_mode::ReadOnly > dvro = make_host_view< access_mode::ReadOnly >(ds);
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
    GRIDTOOLS_STATIC_ASSERT((is_data_view< decltype(devv) >::value), "is_data_view check failed");
    EXPECT_TRUE(check_consistency(ds, devv));
    EXPECT_FALSE(check_consistency(ds, dv));
    EXPECT_FALSE(check_consistency(ds, dvro));
    EXPECT_TRUE(devv.valid());
    EXPECT_FALSE(dv.valid());
    EXPECT_FALSE(dvro.valid());
    mul2<<< 1, 1 >>>(devv);

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
    data_view< data_store_t > dv_tmp = make_host_view< access_mode::ReadWrite >(ds_tmp);
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
