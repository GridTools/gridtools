/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include "common/data_store.hpp"
#include "storage_cuda/data_view_helpers.hpp"
#include "storage_cuda/storage.hpp"
#include "storage_cuda/storage_info.hpp"

using namespace gridtools;

template <typename View>
__global__
void mul2(View s) {
    s(0,0,0) *= 2;
    s(1,0,0) *= 2;    
}

TEST(DataViewTest, Simple) {
    typedef cuda_storage_info<0, layout_map<2,1,0> > storage_info_t;
    typedef data_store< cuda_storage<double>, storage_info_t> data_store_t;
    // create and allocate a data_store
    constexpr storage_info_t si(3,3,3);
    data_store_t ds(si);
    ds.allocate();
    // create a rw view and fill with some data
    data_view<data_store_t> dv = make_host_view(ds);
    static_assert(is_data_view<decltype(dv)>::value, "is_data_view check failed");
    dv(0,0,0) = 50;
    dv(1,0,0) = 60;
    // check if the user protections are working
    static_assert(si.index(1,0,0) == 1, "constexpr index method call failed");    
#ifndef NDEBUG
    std::cout << "Execute death tests.\n";
    ASSERT_DEATH(si.index(0,0,3), "Error triggered");
    ASSERT_DEATH(si.index(0,3,0), "Error triggered");
    ASSERT_DEATH(si.index(3,0,0), "Error triggered");
    ASSERT_DEATH(si.index(5,5,5), "Error triggered");
#endif
    ASSERT_TRUE(si.index(1,0,1) == 97);
    // check if data is there
    EXPECT_EQ(50, dv(0,0,0));
    EXPECT_EQ(dv(1,0,0), 60);
    // create a ro view
    data_view<data_store_t, true> dvro = make_host_view<true>(ds);
    // check if data is the same
    EXPECT_EQ(50, dvro(0,0,0));
    EXPECT_EQ(dvro(1,0,0), 60);
    // views are valid (ds <--> dv and ds <--> dvro) 
    EXPECT_TRUE(valid(ds,dv));
    EXPECT_TRUE(valid(ds,dvro));
    EXPECT_TRUE(dv.valid());
    EXPECT_TRUE(dvro.valid());
    

    // sync, create a device view and call kernel
    ds.sync();
    auto devv = make_device_view(ds);
    static_assert(is_data_view<decltype(devv)>::value, "is_data_view check failed");
    EXPECT_TRUE(valid(ds,devv));
    EXPECT_FALSE(valid(ds,dv));
    EXPECT_FALSE(valid(ds,dvro));
    EXPECT_TRUE(devv.valid());
    EXPECT_FALSE(dv.valid());
    EXPECT_FALSE(dvro.valid());        
    mul2<<<1,1>>>(devv);

    // sync and check if read only host view is valid
    ds.sync();
    EXPECT_FALSE(valid(ds,devv));
    EXPECT_FALSE(valid(ds,dv));
    EXPECT_TRUE(valid(ds,dvro));  
    EXPECT_FALSE(devv.valid());
    EXPECT_FALSE(dv.valid());
    EXPECT_TRUE(dvro.valid());      
    // check if data is the same
    EXPECT_EQ(100, dvro(0,0,0));
    EXPECT_EQ(dvro(1,0,0), 120);

    // create and allocate a second storage
    data_store_t ds_tmp(si);
    ds_tmp.allocate();
    // again create a view
    data_view<data_store_t> dv_tmp = make_host_view(ds_tmp);
    // the combination ds_tmp <--> dv/dvro is not a valid view
    EXPECT_FALSE(valid(ds,dv_tmp));
    EXPECT_FALSE(valid(ds_tmp,devv));
    EXPECT_FALSE(valid(ds_tmp,dvro));
    EXPECT_TRUE(valid(ds_tmp,dv_tmp));

    EXPECT_TRUE(dv_tmp.valid());
    EXPECT_FALSE(devv.valid());
    EXPECT_TRUE(dvro.valid());
    EXPECT_TRUE(dv_tmp.valid());    

    // destroy a storage, this should also invalidate the views
    ds.free();
    EXPECT_FALSE(valid(ds,dv));
    EXPECT_FALSE(valid(ds,dvro)); 
}
