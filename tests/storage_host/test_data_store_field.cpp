/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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

TEST(DataStoreFieldTest, Cycle) {
    typedef host_storage_info< 0, layout_map< 2, 1, 0 > > storage_info_t;
    storage_info_t si(3, 3, 3);
    data_store_field< data_store< host_storage< double >, storage_info_t >, 5, 5, 5 > f(si);
    f.allocate();
    // extract ptrs
    double *ptrs_old[] = {f.m_field[0].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[1].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[2].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[3].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[4].get_storage_ptr()->get_cpu_ptr()};
    // shift by -1
    cycle< 0 >::by< -1 >(f);
    // extract ptrs again
    double *ptrs_new[] = {f.m_field[0].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[1].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[2].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[3].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[4].get_storage_ptr()->get_cpu_ptr()};
    // check correct shift (-1)
    ASSERT_TRUE((ptrs_old[0] == ptrs_new[4]));
    ASSERT_TRUE((ptrs_old[1] == ptrs_new[0]));
    ASSERT_TRUE((ptrs_old[2] == ptrs_new[1]));
    ASSERT_TRUE((ptrs_old[3] == ptrs_new[2]));
    ASSERT_TRUE((ptrs_old[4] == ptrs_new[3]));
    // shift again by -5 (no change)
    cycle< 0 >::by< -5 >(f);
    ASSERT_TRUE((ptrs_old[0] == ptrs_new[4]));
    ASSERT_TRUE((ptrs_old[1] == ptrs_new[0]));
    ASSERT_TRUE((ptrs_old[2] == ptrs_new[1]));
    ASSERT_TRUE((ptrs_old[3] == ptrs_new[2]));
    ASSERT_TRUE((ptrs_old[4] == ptrs_new[3]));
    // shift back to normal
    cycle< 0 >::by< 1 >(f);
    double *ptrs_new_1[] = {f.m_field[0].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[1].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[2].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[3].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[4].get_storage_ptr()->get_cpu_ptr()};
    ASSERT_TRUE((ptrs_old[0] == ptrs_new_1[0]));
    ASSERT_TRUE((ptrs_old[1] == ptrs_new_1[1]));
    ASSERT_TRUE((ptrs_old[2] == ptrs_new_1[2]));
    ASSERT_TRUE((ptrs_old[3] == ptrs_new_1[3]));
    ASSERT_TRUE((ptrs_old[4] == ptrs_new_1[4]));
    // shift by -6 (again like before)
    cycle< 0 >::by< -6 >(f);
    double *ptrs_new_2[] = {f.m_field[0].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[1].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[2].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[3].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[4].get_storage_ptr()->get_cpu_ptr()};
    // check correct shift (-6)
    ASSERT_TRUE((ptrs_old[0] == ptrs_new_2[4]));
    ASSERT_TRUE((ptrs_old[1] == ptrs_new_2[0]));
    ASSERT_TRUE((ptrs_old[2] == ptrs_new_2[1]));
    ASSERT_TRUE((ptrs_old[3] == ptrs_new_2[2]));
    ASSERT_TRUE((ptrs_old[4] == ptrs_new_2[3]));
    // shift back to normal (2*5 (no effect) + 6)
    cycle< 0 >::by< 16 >(f);
    double *ptrs_new_3[] = {f.m_field[0].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[1].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[2].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[3].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[4].get_storage_ptr()->get_cpu_ptr()};
    ASSERT_TRUE((ptrs_old[0] == ptrs_new_3[0]));
    ASSERT_TRUE((ptrs_old[1] == ptrs_new_3[1]));
    ASSERT_TRUE((ptrs_old[2] == ptrs_new_3[2]));
    ASSERT_TRUE((ptrs_old[3] == ptrs_new_3[3]));
    ASSERT_TRUE((ptrs_old[4] == ptrs_new_3[4]));
}

TEST(DataStoreFieldTest, CycleAll) {
    typedef host_storage_info< 0, layout_map< 2, 1, 0 > > storage_info_t;
    storage_info_t si(3, 3, 3);
    data_store_field< data_store< host_storage< double >, storage_info_t >, 3, 3, 3 > f(si);
    f.allocate();
    // extract ptrs
    double *ptrs_old[] = {f.m_field[0].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[1].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[2].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[3].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[4].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[5].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[6].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[7].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[8].get_storage_ptr()->get_cpu_ptr()};
    cycle_all::by< -1 >(f);
    double *ptrs_new[] = {f.m_field[0].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[1].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[2].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[3].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[4].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[5].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[6].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[7].get_storage_ptr()->get_cpu_ptr(),
        f.m_field[8].get_storage_ptr()->get_cpu_ptr()};
    // check correct shift (-1)
    // component 0
    ASSERT_TRUE((ptrs_old[0] == ptrs_new[2]));
    ASSERT_TRUE((ptrs_old[1] == ptrs_new[0]));
    ASSERT_TRUE((ptrs_old[2] == ptrs_new[1]));
    // component 1
    ASSERT_TRUE((ptrs_old[3] == ptrs_new[5]));
    ASSERT_TRUE((ptrs_old[4] == ptrs_new[3]));
    ASSERT_TRUE((ptrs_old[5] == ptrs_new[4]));
    // component 2
    ASSERT_TRUE((ptrs_old[6] == ptrs_new[8]));
    ASSERT_TRUE((ptrs_old[7] == ptrs_new[6]));
    ASSERT_TRUE((ptrs_old[8] == ptrs_new[7]));
}
