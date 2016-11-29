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
#include "storage_host/storage.hpp"
#include "storage_host/storage_info.hpp"

using namespace gridtools;

typedef host_storage_info<0, layout_map<2,1,0> > storage_info_t;

void invalid_copy() {
    storage_info_t si(3,3,3);
    data_store<host_storage<double>, storage_info_t> ds1(si);
    data_store<host_storage<double>, storage_info_t> ds2 = ds1;
}

void invalid_copy_assign() {
    storage_info_t si(3,3,3);
    data_store<host_storage<double>, storage_info_t> ds1(si);
    data_store<host_storage<double>, storage_info_t> ds2(si);
    ds2 = ds1;
}

TEST(DataStoreTest, Simple) {
    storage_info_t si(3,3,3);
    // create unallocated data_store
    data_store< host_storage<double>, storage_info_t > ds(si);
    // try to copy and get_storage -> should fail
#ifdef DEBUG
    ASSERT_DEATH(ds.get_storage_ptr(), "data_store is in a non-initialized state.");
    ASSERT_DEATH(invalid_copy(), "Cannot copy a non-initialized data_store.");
    ASSERT_DEATH(invalid_copy_assign(), "Cannot copy a non-initialized data_store.");
#endif
    // allocate space
    ds.allocate(); 
    data_store< host_storage<double>, storage_info_t > ds_tmp_1(si);
    data_store< host_storage<double>, storage_info_t > ds_tmp_2 = ds; // copy construct
    ds_tmp_1 = ds; // copy assign
    data_store< host_storage<double>, storage_info_t > ds1(si);
    ds1.allocate();
    ds1.free(); // destroy the data_store
#ifdef DEBUG
    ASSERT_DEATH(ds1.get_storage_ptr(), "data_store is in a non-initialized state.");
#endif

    // create a copy of a data_store and check equivalence
    data_store< host_storage<double>, storage_info_t > datast(si);
    datast.allocate();
    data_store< host_storage<double>, storage_info_t > datast_cpy(datast); 
    EXPECT_EQ(datast.get_storage_info_ptr(), datast_cpy.get_storage_info_ptr());
    EXPECT_EQ(datast.get_storage_ptr(), datast_cpy.get_storage_ptr());
    // modify the data and check if the copy can see the changes
    datast.get_storage_ptr()->get_cpu_ptr()[0] = 100;
    datast.get_storage_ptr()->get_cpu_ptr()[1] = 200;

    EXPECT_EQ((datast.get_storage_ptr()->get_cpu_ptr()[0]), 100);
    EXPECT_EQ((datast.get_storage_ptr()->get_cpu_ptr()[1]), 200);

    EXPECT_EQ((datast_cpy.get_storage_ptr()->get_cpu_ptr()[0]), 100);
    EXPECT_EQ((datast_cpy.get_storage_ptr()->get_cpu_ptr()[1]), 200);    
}
