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

#include "storage/data_store.hpp"
#include "common/variadic_pack_metafunctions.hpp"
#include "storage/storage_cuda/storage.hpp"
#include "storage/storage_cuda/storage_info.hpp"

using namespace gridtools;

typedef cuda_storage_info< 0, layout_map< 2, 1, 0 > > storage_info_t;

__global__ void mul2(double *s) {
    s[0] *= 2.0;
    s[1] *= 2.0;
}

template < typename StorageInfo >
__global__ void check_vals(double *s, StorageInfo const *si) {
    for (uint_t i = 0; i < 128; ++i)
        for (uint_t j = 0; j < 128; ++j)
            for (uint_t k = 0; k < 80; ++k) {
                int x = si->index(i, j, k);
                if (s[x] > 3.141499 && s[x] < 3.141501) {
                    s[x] = 1.0;
                } else {
                    s[x] = 0.0;
                }
            }
}

TEST(DataStoreTest, Simple) {
    using data_store_t = data_store< cuda_storage< double >, storage_info_t >;
    storage_info_t si(3, 3, 3);
    constexpr storage_info_interface< 0, layout_map< 2, 1, 0 > > csi(3, 3, 3);
    constexpr storage_info_interface< 1, layout_map< 2, 1, 0 >, halo< 2, 1, 0 > > csih(3, 3, 3);
    constexpr storage_info_interface< 2, layout_map< 2, 1, 0 >, halo< 2, 1, 0 >, alignment< 16 > > csiha(3, 3, 3);
    // check sizes, strides, and alignment
    static_assert(csi.dim< 0 >() == 3, "dimension check failed.");
    static_assert(csi.dim< 1 >() == 3, "dimension check failed.");
    static_assert(csi.dim< 2 >() == 3, "dimension check failed.");
    static_assert(csi.unaligned_dim< 0 >() == 3, "dimension check failed.");
    static_assert(csi.unaligned_dim< 1 >() == 3, "dimension check failed.");
    static_assert(csi.unaligned_dim< 2 >() == 3, "dimension check failed.");
    static_assert(csi.stride< 0 >() == 1, "stride check failed.");
    static_assert(csi.stride< 1 >() == 3, "stride check failed.");
    static_assert(csi.stride< 2 >() == 9, "stride check failed.");
    static_assert(csi.unaligned_stride< 0 >() == 1, "stride check failed.");
    static_assert(csi.unaligned_stride< 1 >() == 3, "stride check failed.");
    static_assert(csi.unaligned_stride< 2 >() == 9, "stride check failed.");
    static_assert(csi.get_initial_offset() == 0, "init. offset check failed");

    static_assert(csih.dim< 0 >() == 7, "dimension check failed.");
    static_assert(csih.dim< 1 >() == 5, "dimension check failed.");
    static_assert(csih.dim< 2 >() == 3, "dimension check failed.");
    static_assert(csih.unaligned_dim< 0 >() == 7, "dimension check failed.");
    static_assert(csih.unaligned_dim< 1 >() == 5, "dimension check failed.");
    static_assert(csih.unaligned_dim< 2 >() == 3, "dimension check failed.");
    static_assert(csih.stride< 0 >() == 1, "stride check failed.");
    static_assert(csih.stride< 1 >() == 7, "stride check failed.");
    static_assert(csih.stride< 2 >() == 35, "stride check failed.");
    static_assert(csih.unaligned_stride< 0 >() == 1, "stride check failed.");
    static_assert(csih.unaligned_stride< 1 >() == 7, "stride check failed.");
    static_assert(csih.unaligned_stride< 2 >() == 35, "stride check failed.");
    static_assert(csih.get_initial_offset() == 0, "init. offset check failed");

    EXPECT_EQ(csiha.dim< 0 >(), 16);
    EXPECT_EQ(csiha.dim< 1 >(), 5);
    EXPECT_EQ(csiha.dim< 2 >(), 3);
    static_assert(csiha.unaligned_dim< 0 >() == 7, "dimension check failed.");
    static_assert(csiha.unaligned_dim< 1 >() == 5, "dimension check failed.");
    static_assert(csiha.unaligned_dim< 2 >() == 3, "dimension check failed.");
    EXPECT_EQ(csiha.stride< 0 >(), 1);
    EXPECT_EQ(csiha.stride< 1 >(), 16);
    EXPECT_EQ(csiha.stride< 2 >(), 80);
    static_assert(csiha.unaligned_stride< 0 >() == 1, "stride check failed.");
    static_assert(csiha.unaligned_stride< 1 >() == 7, "stride check failed.");
    static_assert(csiha.unaligned_stride< 2 >() == 35, "stride check failed.");
    static_assert(csiha.get_initial_offset() == 14, "init. offset check failed");

    // create unallocated data_store
    data_store_t ds;
    // allocate space
    ds.allocate(si);
    data_store_t ds_tmp_1(si);
    data_store_t ds_tmp_2 = ds; // copy construct
    ds_tmp_1 = ds;              // copy assign
    data_store_t ds1;
    ds1.allocate(si);
    ds1.reset(); // destroy the data_store

    // create a copy of a data_store and check equivalence
    data_store_t datast;
    datast.allocate(si);
    data_store_t datast_cpy(datast);
    EXPECT_EQ(datast.get_storage_info_ptr().get(), datast_cpy.get_storage_info_ptr().get());
    EXPECT_EQ(datast.get_storage_ptr().get(), datast_cpy.get_storage_ptr().get());

    // modify the data and check if the copy can see the changes
    datast.get_storage_ptr()->get_cpu_ptr()[0] = 100;
    datast.get_storage_ptr()->get_cpu_ptr()[1] = 200;
    EXPECT_EQ((datast.get_storage_ptr()->get_cpu_ptr()[0]), 100);
    EXPECT_EQ((datast.get_storage_ptr()->get_cpu_ptr()[1]), 200);
    EXPECT_EQ((datast_cpy.get_storage_ptr()->get_cpu_ptr()[0]), 100);
    EXPECT_EQ((datast_cpy.get_storage_ptr()->get_cpu_ptr()[1]), 200);

    // clone to device
    datast.clone_to_device();
    mul2<<< 1, 1 >>>(datast.get_storage_ptr()->get_gpu_ptr());

    // check again
    datast.get_storage_ptr()->get_cpu_ptr()[0] = 200;
    datast.get_storage_ptr()->get_cpu_ptr()[1] = 400;
    EXPECT_EQ((datast.get_storage_ptr()->get_cpu_ptr()[0]), 200);
    EXPECT_EQ((datast.get_storage_ptr()->get_cpu_ptr()[1]), 400);
    EXPECT_EQ((datast_cpy.get_storage_ptr()->get_cpu_ptr()[0]), 200);
    EXPECT_EQ((datast_cpy.get_storage_ptr()->get_cpu_ptr()[1]), 400);

    // test some copy assignment operations
    data_store< cuda_storage< double >, storage_info_t > ds_cpy_ass1(si);
    data_store< cuda_storage< double >, storage_info_t > ds_cpy_ass2;
    ds_cpy_ass2 = ds_cpy_ass1;
    ASSERT_TRUE(ds_cpy_ass2.get_storage_ptr()->get_cpu_ptr() == ds_cpy_ass1.get_storage_ptr()->get_cpu_ptr());
    ASSERT_TRUE(ds_cpy_ass2.get_storage_ptr()->get_gpu_ptr() == ds_cpy_ass1.get_storage_ptr()->get_gpu_ptr());
    ASSERT_TRUE(*ds_cpy_ass2.get_storage_info_ptr() == *ds_cpy_ass1.get_storage_info_ptr());
}

TEST(DataStoreTest, States) {
    using data_store_t = data_store< cuda_storage< double >, storage_info_t >;
    storage_info_t si(3, 3, 3);
    // create and allocate data_store
    data_store_t ds(si);

    // intial state should be
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu);

    // host write views should be valid, this means the device needs an update
    ds.reactivate_host_write_views();
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu);
    EXPECT_TRUE(ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu);

    // synchronize everything, valid on both sides
    ds.sync();
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu);

    // device write views should be valid, this means the host needs an update
    ds.reactivate_device_write_views();
    EXPECT_TRUE(ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu);

    // go back from device
    ds.sync();
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu);
    EXPECT_FALSE(ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu);
}

TEST(DataStoreTest, Initializer) {
    storage_info_t si(128, 128, 80);
    data_store< cuda_storage< double >, storage_info_t > ds(si, 3.1415);
    check_vals<<< 1, 1 >>>(ds.get_storage_ptr()->get_gpu_ptr(), ds.get_storage_info_ptr()->get_gpu_ptr());
    ds.clone_from_device();
    for (uint_t i = 0; i < 128; ++i)
        for (uint_t j = 0; j < 128; ++j)
            for (uint_t k = 0; k < 80; ++k)
                EXPECT_EQ((ds.get_storage_ptr()->get_cpu_ptr()[si.index(i, j, k)]), 1.0);
}

TEST(DataStoreTest, LambdaInitializer) {
    storage_info_t si(10, 11, 12);
    data_store< cuda_storage< double >, storage_info_t > ds(si, [](int i, int j, int k) { return i + j + k; });
    for (uint_t i = 0; i < 10; ++i)
        for (uint_t j = 0; j < 11; ++j)
            for (uint_t k = 0; k < 12; ++k)
                EXPECT_EQ((ds.get_storage_ptr()->get_cpu_ptr()[si.index(i, j, k)]), (i + j + k));
}

TEST(DataStoreTest, Naming) {
    storage_info_t si(10, 11, 12);
    // no naming
    data_store< cuda_storage< double >, storage_info_t > ds1_nn;
    data_store< cuda_storage< double >, storage_info_t > ds2_nn(si);
    data_store< cuda_storage< double >, storage_info_t > ds3_nn(si, 1.0);
    data_store< cuda_storage< double >, storage_info_t > ds4_nn(si, [](int i, int j, int k) { return i + j + k; });
    EXPECT_EQ(ds1_nn.name(), "");
    EXPECT_EQ(ds2_nn.name(), "");
    EXPECT_EQ(ds3_nn.name(), "");
    EXPECT_EQ(ds4_nn.name(), "");

    // test naming
    data_store< cuda_storage< double >, storage_info_t > ds1("empty storage");
    data_store< cuda_storage< double >, storage_info_t > ds2(si, "standard storage");
    data_store< cuda_storage< double >, storage_info_t > ds3(si, 1.0, "value init. storage");
    data_store< cuda_storage< double >, storage_info_t > ds4(
        si, [](int i, int j, int k) { return i + j + k; }, "lambda init. storage");
    EXPECT_EQ(ds1.name(), "empty storage");
    EXPECT_EQ(ds2.name(), "standard storage");
    EXPECT_EQ(ds3.name(), "value init. storage");
    EXPECT_EQ(ds4.name(), "lambda init. storage");

    // create a copy and see if still ok
    auto ds2_tmp = ds2;
    EXPECT_EQ(ds2_tmp.name(), "standard storage");
    ds1 = ds3;
    EXPECT_EQ(ds1.name(), "value init. storage");
    EXPECT_EQ(ds3.name(), "value init. storage");
}

TEST(DataStoreTest, ExternalPointer) {
    // test with an external CPU pointer
    storage_info_t si(10, 10, 10);
    double *external_ptr = new double[si.size()];
    // create a data_store with externally managed storage
    data_store< cuda_storage< double >, storage_info_t > ds(si, external_ptr, ownership::ExternalCPU);
    ds.sync();
    // create a copy (double free checks)
    data_store< cuda_storage< double >, storage_info_t > ds_cpy = ds;
    // check values
    for (uint_t i = 0; i < 10; ++i)
        for (uint_t j = 0; j < 10; ++j)
            for (uint_t k = 0; k < 10; ++k) {
                external_ptr[si.index(i, j, k)] = 3.1415;
                EXPECT_EQ((ds.get_storage_ptr()->get_cpu_ptr()[si.index(i, j, k)]), 3.1415);
                EXPECT_EQ((ds_cpy.get_storage_ptr()->get_cpu_ptr()[si.index(i, j, k)]), 3.1415);
            }
    // delete the ptr
    delete[] external_ptr;
}

TEST(DataStoreTest, DimAndSizeInterface) {
    storage_info_t si(128, 128, 80);
    data_store< cuda_storage< double >, storage_info_t > ds(si, 3.1415);
    ASSERT_TRUE((ds.size() == si.size()));
    ASSERT_TRUE((ds.dim< 0 >() == si.dim< 0 >()));
    ASSERT_TRUE((ds.dim< 1 >() == si.dim< 1 >()));
    ASSERT_TRUE((ds.dim< 2 >() == si.dim< 2 >()));
}

TEST(DataStoreTest, ExternalGPUPointer) {
    // test with an external GPU pointer
    storage_info_t si(10, 10, 10);
    double *external_gpu_ptr;
    double *external_cpu_ptr = new double[si.size()];
    // initialize CPU ptr
    for (uint_t i = 0; i < si.size(); ++i) {
        external_cpu_ptr[i] = 3.1415;
    }
    // create a GPU ptr
    cudaError_t err = cudaMalloc(&external_gpu_ptr, si.size() * sizeof(double));
    ASSERT_TRUE((err == cudaSuccess));
    // initialize the GPU ptr
    err = cudaMemcpy(
        (void *)external_gpu_ptr, (void *)external_cpu_ptr, si.size() * sizeof(double), cudaMemcpyHostToDevice);
    ASSERT_TRUE((err == cudaSuccess));
    // create a data_store with externally managed storage
    data_store< cuda_storage< double >, storage_info_t > ds(si, external_gpu_ptr, ownership::ExternalGPU);
    ds.sync();
    // create some copies
    data_store< cuda_storage< double >, storage_info_t > ds_cpy_1(ds);
    data_store< cuda_storage< double >, storage_info_t > ds_cpy_2 = ds_cpy_1;
    EXPECT_EQ(ds_cpy_1.get_storage_ptr()->get_cpu_ptr(), ds_cpy_2.get_storage_ptr()->get_cpu_ptr());
    EXPECT_EQ(ds_cpy_2.get_storage_ptr()->get_cpu_ptr(), ds.get_storage_ptr()->get_cpu_ptr());
    // create a copy (double free checks)
    data_store< cuda_storage< double >, storage_info_t > ds_cpy = ds;
    // check values
    for (uint_t i = 0; i < 10; ++i)
        for (uint_t j = 0; j < 10; ++j)
            for (uint_t k = 0; k < 10; ++k) {
                EXPECT_EQ((ds.get_storage_ptr()->get_cpu_ptr()[si.index(i, j, k)]), 3.1415);
                EXPECT_EQ((ds_cpy.get_storage_ptr()->get_cpu_ptr()[si.index(i, j, k)]), 3.1415);
            }
    // delete the ptr
    delete[] external_cpu_ptr;
    cudaFree(external_gpu_ptr);
}
