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
#include "storage_cuda/storage.hpp"
#include "storage_cuda/storage_info.hpp"

using namespace gridtools;

typedef cuda_storage_info<0, layout_map<2,1,0> > storage_info_t;

void invalid_copy() {
    storage_info_t si(3,3,3);
    data_store<cuda_storage<double>, storage_info_t> ds1(si);
    data_store<cuda_storage<double>, storage_info_t> ds2 = ds1;
}

void invalid_copy_assign() {
    storage_info_t si(3,3,3);
    data_store<cuda_storage<double>, storage_info_t> ds1(si);
    data_store<cuda_storage<double>, storage_info_t> ds2(si);
    ds2 = ds1;
}

__global__
void mul2(double* s) {
    s[0] *= 2.0;
    s[1] *= 2.0;
}

TEST(DataStoreTest, Simple) {
    using data_store_t = data_store< cuda_storage<double>, storage_info_t >;
    storage_info_t si(3,3,3);
    constexpr storage_info_interface<0, layout_map<2,1,0> > csi(3,3,3);
    constexpr storage_info_interface<1, layout_map<2,1,0>, halo<2,1,0> > csih(3,3,3);
    constexpr storage_info_interface<2, layout_map<2,1,0>, halo<2,1,0>, alignment<16> > csiha(3,3,3);
    // check sizes, strides, and alignment
    static_assert(csi.dim<0>() == 3, "dimension check failed.");
    static_assert(csi.dim<1>() == 3, "dimension check failed.");
    static_assert(csi.dim<2>() == 3, "dimension check failed.");
    static_assert(csi.unaligned_dim<0>() == 3, "dimension check failed.");
    static_assert(csi.unaligned_dim<1>() == 3, "dimension check failed.");
    static_assert(csi.unaligned_dim<2>() == 3, "dimension check failed.");
    static_assert(csi.stride<0>() == 1, "stride check failed.");
    static_assert(csi.stride<1>() == 3, "stride check failed.");
    static_assert(csi.stride<2>() == 9, "stride check failed.");
    static_assert(csi.unaligned_stride<0>() == 1, "stride check failed.");
    static_assert(csi.unaligned_stride<1>() == 3, "stride check failed.");
    static_assert(csi.unaligned_stride<2>() == 9, "stride check failed.");
    static_assert(csi.get_initial_offset() == 0, "init. offset check failed");

    static_assert(csih.dim<0>() == 7, "dimension check failed.");
    static_assert(csih.dim<1>() == 5, "dimension check failed.");
    static_assert(csih.dim<2>() == 3, "dimension check failed.");
    static_assert(csih.unaligned_dim<0>() == 7, "dimension check failed.");
    static_assert(csih.unaligned_dim<1>() == 5, "dimension check failed.");
    static_assert(csih.unaligned_dim<2>() == 3, "dimension check failed.");
    static_assert(csih.stride<0>() == 1, "stride check failed.");
    static_assert(csih.stride<1>() == 7, "stride check failed.");
    static_assert(csih.stride<2>() == 35, "stride check failed.");
    static_assert(csih.unaligned_stride<0>() == 1, "stride check failed.");
    static_assert(csih.unaligned_stride<1>() == 7, "stride check failed.");
    static_assert(csih.unaligned_stride<2>() == 35, "stride check failed.");
    static_assert(csih.get_initial_offset() == 0, "init. offset check failed");
    
    EXPECT_EQ(csiha.dim<0>(), 16); 
    EXPECT_EQ(csiha.dim<1>(), 5); 
    EXPECT_EQ(csiha.dim<2>(), 3); 
    static_assert(csiha.unaligned_dim<0>() == 7, "dimension check failed.");
    static_assert(csiha.unaligned_dim<1>() == 5, "dimension check failed.");
    static_assert(csiha.unaligned_dim<2>() == 3, "dimension check failed.");
    EXPECT_EQ(csiha.stride<0>(), 1);
    EXPECT_EQ(csiha.stride<1>(), 16);
    EXPECT_EQ(csiha.stride<2>(), 80);
    static_assert(csiha.unaligned_stride<0>() == 1, "stride check failed.");
    static_assert(csiha.unaligned_stride<1>() == 7, "stride check failed.");
    static_assert(csiha.unaligned_stride<2>() == 35, "stride check failed.");
    static_assert(csiha.get_initial_offset() == 14, "init. offset check failed");

    // create unallocated data_store
    data_store_t ds(si);
    // try to copy and get_storage -> should fail
    ASSERT_DEATH(ds.get_storage_ptr(), "data_store is in a non-initialized state.");
#ifndef __CUDACC__
    // death tests that call cudaMalloc, etc. do not work
    ASSERT_DEATH(invalid_copy(), "Cannot copy a non-initialized data_store.");
    ASSERT_DEATH(invalid_copy_assign(), "Cannot copy a non-initialized data_store.");
#endif
    // allocate space
    ds.allocate(); 
    data_store_t ds_tmp_1(si);
    data_store_t ds_tmp_2 = ds; // copy construct
    ds_tmp_1 = ds; // copy assign
    data_store_t ds1(si);
    ds1.allocate();
    ds1.free(); // destroy the data_store
    ASSERT_DEATH(ds1.get_storage_ptr(), "data_store is in a non-initialized state.");

    // create a copy of a data_store and check equivalence
    data_store_t datast(si);
    datast.allocate();
    data_store_t datast_cpy(datast); 
    EXPECT_EQ(datast.get_storage_info_ptr(), datast_cpy.get_storage_info_ptr());
    EXPECT_EQ(datast.get_storage_ptr(), datast_cpy.get_storage_ptr());

    // modify the data and check if the copy can see the changes
    datast.get_storage_ptr()->get_cpu_ptr()[0] = 100;
    datast.get_storage_ptr()->get_cpu_ptr()[1] = 200;
    EXPECT_EQ((datast.get_storage_ptr()->get_cpu_ptr()[0]), 100);
    EXPECT_EQ((datast.get_storage_ptr()->get_cpu_ptr()[1]), 200);
    EXPECT_EQ((datast_cpy.get_storage_ptr()->get_cpu_ptr()[0]), 100);
    EXPECT_EQ((datast_cpy.get_storage_ptr()->get_cpu_ptr()[1]), 200);
    
    // clone to device
    datast.clone_to_device();
    mul2<<<1,1>>>(datast.get_storage_ptr()->get_gpu_ptr());
    
    // check again    
    datast.get_storage_ptr()->get_cpu_ptr()[0] = 200;
    datast.get_storage_ptr()->get_cpu_ptr()[1] = 400;
    EXPECT_EQ((datast.get_storage_ptr()->get_cpu_ptr()[0]), 200);
    EXPECT_EQ((datast.get_storage_ptr()->get_cpu_ptr()[1]), 400);
    EXPECT_EQ((datast_cpy.get_storage_ptr()->get_cpu_ptr()[0]), 200);
    EXPECT_EQ((datast_cpy.get_storage_ptr()->get_cpu_ptr()[1]), 400);
}

TEST(DataStoreTest, States) {
    using data_store_t = data_store< cuda_storage<double>, storage_info_t >;
    storage_info_t si(3,3,3);
    // create and allocate data_store
    data_store_t ds(si);
    ds.allocate();

    // intial state should be 
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_od);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu);
    EXPECT_TRUE(ds.is_on_host());
    EXPECT_FALSE(ds.is_on_device());
    
    // host write views should be valid, this means the device needs an update
    ds.reactivate_host_write_views();
    EXPECT_TRUE(ds.is_on_host());
    EXPECT_FALSE(ds.is_on_device());
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_od);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu);
    EXPECT_TRUE(ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu);

    // synchronize everything, valid on both sides
    ds.sync();
    EXPECT_TRUE(ds.get_storage_ptr()->get_state_machine_ptr()->m_od);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu);
    EXPECT_FALSE(ds.is_on_host());
    EXPECT_TRUE(ds.is_on_device());

    // device write views should be valid, this means the host needs an update
    ds.reactivate_device_write_views();
    EXPECT_FALSE(ds.is_on_host());
    EXPECT_TRUE(ds.is_on_device());
    EXPECT_TRUE(ds.get_storage_ptr()->get_state_machine_ptr()->m_od);
    EXPECT_TRUE(ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu);

    // go back from device
    ds.sync();
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_od);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu);
    EXPECT_TRUE(ds.is_on_host());
    EXPECT_FALSE(ds.is_on_device());
}
