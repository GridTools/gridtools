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
#include <storage/storage_host/data_view_helpers.hpp>
#include <storage/storage_host/host_storage.hpp>
#include <storage/storage_host/host_storage_info.hpp>

using namespace gridtools;

TEST(DataViewTest, Simple) {
    typedef host_storage_info< 0, layout_map< 2, 1, 0 > > storage_info_t;
    typedef data_store< host_storage< double >, storage_info_t > data_store_t;
    // create and allocate a data_store
    constexpr storage_info_t si(3, 3, 3);
    data_store_t ds;
    ds.allocate(si);
    // create a rw view and fill with some data
    data_view< data_store_t > dv = make_host_view(ds);
    EXPECT_TRUE(dv.valid());
    GRIDTOOLS_STATIC_ASSERT(is_data_view< decltype(dv) >::value, "is_data_view check failed");
    dv(0, 0, 0) = 50;
    dv(0, 0, 1) = 60;

    // check if dim interface works
    ASSERT_TRUE((si.dim< 0 >() == dv.dim< 0 >()));
    ASSERT_TRUE((si.dim< 1 >() == dv.dim< 1 >()));
    ASSERT_TRUE((si.dim< 2 >() == dv.dim< 2 >()));
    ASSERT_TRUE((si.total_length() == dv.total_length()));
    ASSERT_TRUE((si.padded_total_length() == dv.padded_total_length()));
    ASSERT_TRUE((si.length() == dv.length()));

    // check if data is there
    EXPECT_EQ(50, dv(0, 0, 0));
    EXPECT_EQ(dv(0, 0, 1), 60);
    // check if the user protections are working
    GRIDTOOLS_STATIC_ASSERT(si.index(1, 0, 0) == 1, "constexpr index method call failed");

    std::cout << "Execute death tests.\n";

// this checks are only performed in debug mode
#ifndef NDEBUG
    EXPECT_THROW(si.index(0, 0, 3), std::runtime_error);
    EXPECT_THROW(si.index(0, 3, 0), std::runtime_error);
    EXPECT_THROW(si.index(3, 0, 0), std::runtime_error);
    EXPECT_THROW(si.index(5, 5, 5), std::runtime_error);
#endif

    ASSERT_TRUE(si.index(1, 0, 1) == 10);
    // create a ro view
    data_view< data_store_t, access_mode::ReadOnly > dvro = make_host_view< access_mode::ReadOnly >(ds);
    // check if data is the same
    EXPECT_EQ(50, dvro(0, 0, 0));
    EXPECT_EQ(dvro(0, 0, 1), 60);
    // views are valid (ds <--> dv and ds <--> dvro)
    EXPECT_TRUE(check_consistency(ds, dv));
    EXPECT_TRUE(check_consistency(ds, dvro));

    // create and allocate a second storage
    data_store_t ds_tmp;
    ds_tmp.allocate(si);
    // again create a view
    data_view< data_store_t > dv_tmp = make_host_view< access_mode::ReadWrite >(ds_tmp);
    // the combination ds_tmp <--> dv/dvro is not a valid view
    EXPECT_FALSE(check_consistency(ds, dv_tmp));
    EXPECT_FALSE(check_consistency(ds_tmp, dv));
    EXPECT_FALSE(check_consistency(ds_tmp, dvro));
    EXPECT_TRUE(check_consistency(ds_tmp, dv_tmp));
    EXPECT_TRUE(dv_tmp.valid());

    // destroy a storage, this should also invalidate the views
    ds.reset();
    EXPECT_FALSE(check_consistency(ds, dv));
    EXPECT_FALSE(check_consistency(ds, dvro));
}

TEST(DataViewTest, ZeroSize) {
    typedef host_storage_info< 0, layout_map< 0 > > storage_info_t;
    typedef data_store< host_storage< double >, storage_info_t > data_store_t;
    // create and allocate a data_store
    data_store_t ds;
    data_view< data_store_t, access_mode::ReadOnly > dvro = make_host_view< access_mode::ReadOnly >(ds);
}

TEST(DataViewTest, ArrayAPI) {
    typedef host_storage_info< 0, layout_map< 0, 1, 2 > > storage_info_t;
    storage_info_t si(2, 2, 2);

    typedef data_store< host_storage< double >, storage_info_t > data_store_t;
    // create and allocate a data_store
    data_store_t ds(si);
    auto dvro = make_host_view< access_mode::ReadWrite >(ds);

    dvro({1, 1, 1}) = 2.0;
    EXPECT_TRUE((dvro(array< int, 3 >{(int)1, (int)1, (int)1}) == 2.0));
}
