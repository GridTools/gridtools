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

#include "common/data_store_field.hpp"
#include "storage_host/data_field_view_helpers.hpp"
#include "storage_host/data_view_helpers.hpp"
#include "storage_host/storage.hpp"
#include "storage_host/storage_info.hpp"

using namespace gridtools;

TEST(DataStoreFieldTest, InstantiateAllocateFree) {
    typedef host_storage_info< 0, layout_map< 2, 1, 0 > > storage_info_t;
    storage_info_t si(3, 3, 3);
    // create unallocated data_store_field
    data_store_field< data_store< host_storage< double >, storage_info_t >, 1, 1, 1 > f(si);
    // check if valid
    EXPECT_FALSE(f.valid());
    EXPECT_FALSE((f.get< 0, 0 >().valid()));
    EXPECT_FALSE((f.get< 1, 0 >().valid()));
    EXPECT_FALSE((f.get< 2, 0 >().valid()));
    // allocate one storage and make it valid
    f.get< 0, 0 >().allocate();
    EXPECT_TRUE((f.get< 0, 0 >().valid()));
    EXPECT_FALSE(f.valid());
    EXPECT_FALSE((f.get< 1, 0 >().valid()));
    EXPECT_FALSE((f.get< 2, 0 >().valid()));
    // allocate the other storages
    f.get< 1, 0 >().allocate();
    f.get< 2, 0 >().allocate();
    EXPECT_TRUE(f.valid());
    // free one and see if invalid
    f.get< 1, 0 >().free();
    EXPECT_FALSE(f.valid());
}

TEST(DataStoreFieldTest, FillAndReadData) {
    typedef host_storage_info< 0, layout_map< 2, 1, 0 > > storage_info_t;
    storage_info_t si(3, 3, 3);
    // create unallocated data_store_field
    data_store_field< data_store< host_storage< double >, storage_info_t >, 1, 1, 1 > f(si);
    f.allocate();
    // access the first storage of the first dimension and set the first value to 5
    auto hv = make_field_host_view(f);
    static_assert(is_data_field_view< decltype(hv) >::value, "is_data_field_view is not working anymore");
    hv.get< 0, 0 >()(0, 0, 0) = 5;
    hv.get< 1, 0 >()(0, 0, 0) = -5;
    // manually get the view of the first storage element in the data view (equivalent to get<0,0>...)
    data_store< host_storage< double >, storage_info_t > partial_1 = f.get_field()[0];
    data_store< host_storage< double >, storage_info_t > partial_2 = f.get_field()[1];
    auto hv1 = make_host_view< true >(partial_1); // read only view
    auto hv2 = make_host_view(partial_2);         // read write view (just for fun)
    EXPECT_EQ(hv1(0, 0, 0), 5);
    EXPECT_EQ(hv2(0, 0, 0), -5);
    EXPECT_TRUE(valid(f, hv));
    EXPECT_TRUE(valid(partial_1, hv1));
    EXPECT_TRUE(valid(partial_2, hv2));
    swap< 0, 0 >::with< 1, 0 >(f);
    EXPECT_FALSE(valid(f, hv));
    EXPECT_FALSE(valid(partial_1, hv1));
    EXPECT_FALSE(valid(partial_2, hv2));
    // we have to update the host view because we swapped the ptrs
    hv1 = make_host_view< true >(partial_1);
    hv2 = make_host_view(partial_2);
    EXPECT_EQ(hv1(0, 0, 0), -5);
    EXPECT_EQ(hv2(0, 0, 0), 5);
}

TEST(DataStoreFieldTest, GetSet) {
    typedef host_storage_info< 0, layout_map< 2, 1, 0 > > storage_info_t;
    storage_info_t si(3, 3, 3);
    // create unallocated data_store_field
    data_store_field< data_store< host_storage< double >, storage_info_t >, 1, 1, 1 > f(si);
    f.allocate();
    // get a storage and compare ptrs
    data_store< host_storage< double >, storage_info_t > st = f.get< 1, 0 >();
    EXPECT_EQ(st.get_storage_ptr()->get_cpu_ptr(), f.get_field()[1].get_storage_ptr()->get_cpu_ptr());
    // set a new storage
    data_store< host_storage< double >, storage_info_t > nst(si);
    nst.allocate();
    f.set< 1, 0 >(nst);
    EXPECT_NE(st.get_storage_ptr()->get_cpu_ptr(), f.get_field()[1].get_storage_ptr()->get_cpu_ptr());
    EXPECT_EQ(nst.get_storage_ptr()->get_cpu_ptr(), f.get_field()[1].get_storage_ptr()->get_cpu_ptr());
}

TEST(DataStoreFieldTest, MultiStorageInfo) {
    typedef host_storage_info< 0, layout_map< 2, 1, 0 > > storage_info_t;
    storage_info_t si1(3, 3, 3);
    storage_info_t si2(4, 4, 4);
    storage_info_t si3(5, 5, 5);
    // create unallocated data_store_field
    data_store_field< data_store< host_storage< double >, storage_info_t >, 1, 2, 3 > f(si1, si2, si3);
    f.allocate();
    // check for correct sizes
    data_store< host_storage< double >, storage_info_t > st1 = f.get< 0, 0 >();
    EXPECT_EQ((st1.get_storage_info_ptr()->size()), 3 * 3 * 3);

    data_store< host_storage< double >, storage_info_t > st20 = f.get< 1, 0 >();
    EXPECT_EQ((st20.get_storage_info_ptr()->size()), 4 * 4 * 4);
    data_store< host_storage< double >, storage_info_t > st21 = f.get< 1, 1 >();
    EXPECT_EQ((st21.get_storage_info_ptr()->size()), 4 * 4 * 4);

    data_store< host_storage< double >, storage_info_t > st30 = f.get< 2, 0 >();
    EXPECT_EQ((st30.get_storage_info_ptr()->size()), 5 * 5 * 5);
    data_store< host_storage< double >, storage_info_t > st31 = f.get< 2, 1 >();
    EXPECT_EQ((st31.get_storage_info_ptr()->size()), 5 * 5 * 5);
    data_store< host_storage< double >, storage_info_t > st32 = f.get< 2, 2 >();
    EXPECT_EQ((st32.get_storage_info_ptr()->size()), 5 * 5 * 5);

    auto hv = make_field_host_view(f);
    EXPECT_EQ((hv.get< 0, 0 >().m_storage_info->size()), 3 * 3 * 3);

    EXPECT_EQ((hv.get< 1, 0 >().m_storage_info->size()), 4 * 4 * 4);
    EXPECT_EQ((hv.get< 1, 1 >().m_storage_info->size()), 4 * 4 * 4);

    EXPECT_EQ((hv.get< 2, 0 >().m_storage_info->size()), 5 * 5 * 5);
    EXPECT_EQ((hv.get< 2, 1 >().m_storage_info->size()), 5 * 5 * 5);
    EXPECT_EQ((hv.get< 2, 2 >().m_storage_info->size()), 5 * 5 * 5);
}
