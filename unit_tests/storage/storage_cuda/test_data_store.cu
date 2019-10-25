/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "gtest/gtest.h"

#include <gridtools/common/gt_assert.hpp>
#include <gridtools/common/variadic_pack_metafunctions.hpp>
#include <gridtools/storage/data_store.hpp>
#include <gridtools/storage/storage_cuda/cuda_storage.hpp>
#include <gridtools/storage/storage_cuda/cuda_storage_info.hpp>

using namespace gridtools;

typedef storage_info<0, layout_map<2, 1, 0>, halo<0, 0, 0>, alignment<32>> storage_info_t;

__global__ void mul2(double *s) {
    s[0] *= 2.0;
    s[1] *= 2.0;
}

template <typename StorageInfo>
__global__ void check_vals(double *s, StorageInfo const *si) {
    for (uint_t i = 0; i < si->template total_length<0>(); ++i)
        for (uint_t j = 0; j < si->template total_length<1>(); ++j)
            for (uint_t k = 0; k < si->template total_length<2>(); ++k) {
                int x = si->index(i, j, k);
                if (s[x] > 3.141499 && s[x] < 3.141501) {
                    s[x] = 1.0;
                } else {
                    s[x] = 0.0;
                }
            }
}

template <typename StorageInfo>
__global__ void check_vals_lambda(double *s, StorageInfo const *si) {
    for (uint_t i = 0; i < si->template total_length<0>(); ++i)
        for (uint_t j = 0; j < si->template total_length<1>(); ++j)
            for (uint_t k = 0; k < si->template total_length<2>(); ++k) {
                int x = si->index(i, j, k);
                if (s[x] == i + j + k) {
                    s[x] = i + j + k + 1.0;
                } else {
                    s[x] = 0.0;
                }
            }
}

TEST(DataStoreTest, Simple) {
    using data_store_t = data_store<cuda_storage<double>, storage_info_t>;
    storage_info_t si(3, 3, 3);

    constexpr storage_info<0, layout_map<2, 1, 0>> csi(3, 3, 3);
    constexpr storage_info<1, layout_map<2, 1, 0>, halo<2, 1, 0>> csih(7, 5, 3);
    constexpr storage_info<2, layout_map<2, 1, 0>, halo<2, 1, 0>, alignment<16>> csiha(7, 5, 3);

    // check sizes, strides, and alignment
    EXPECT_EQ(csi.total_length<0>(), 3);
    EXPECT_EQ(csi.total_length<1>(), 3);
    EXPECT_EQ(csi.total_length<2>(), 3);

    EXPECT_EQ(csi.stride<0>(), 1);
    EXPECT_EQ(csi.stride<1>(), 3);
    EXPECT_EQ(csi.stride<2>(), 9);

    EXPECT_EQ(csih.total_length<0>(), 7);
    EXPECT_EQ(csih.total_length<1>(), 5);
    EXPECT_EQ(csih.total_length<2>(), 3);

    EXPECT_EQ(csih.stride<0>(), 1);
    EXPECT_EQ(csih.stride<1>(), 7);
    EXPECT_EQ(csih.stride<2>(), 35);

    EXPECT_EQ(csiha.total_length<0>(), 7);
    EXPECT_EQ(csiha.total_length<1>(), 5);
    EXPECT_EQ(csiha.total_length<2>(), 3);

    EXPECT_EQ(csiha.padded_length<0>(), 16);
    EXPECT_EQ(csiha.padded_length<1>(), 5);
    EXPECT_EQ(csiha.padded_length<2>(), 3);

    EXPECT_EQ(csiha.stride<0>(), 1);
    EXPECT_EQ(csiha.stride<1>(), 16);
    EXPECT_EQ(csiha.stride<2>(), 16 * 5);

    data_store_t ds(si);
    data_store_t ds_tmp_1(si);
    data_store_t ds_tmp_2 = ds; // copy construct
    ds_tmp_1 = ds;              // copy assign

    // create a copy of a data_store and check equivalence
    data_store_t datast(si);
    data_store_t datast_cpy(datast);
    EXPECT_EQ(&datast.info(), &datast_cpy.info());
    EXPECT_EQ(&datast.storage(), &datast_cpy.storage());

    // modify the data and check if the copy can see the changes
    datast.storage().get_cpu_ptr()[0] = 100;
    datast.storage().get_cpu_ptr()[1] = 200;
    EXPECT_EQ(datast.storage().get_cpu_ptr()[0], 100);
    EXPECT_EQ(datast.storage().get_cpu_ptr()[1], 200);
    EXPECT_EQ(datast_cpy.storage().get_cpu_ptr()[0], 100);
    EXPECT_EQ(datast_cpy.storage().get_cpu_ptr()[1], 200);

    // clone to device
    datast.clone_to_device();
    mul2<<<1, 1>>>(datast.storage().get_target_ptr());

    // check again
    datast.storage().get_cpu_ptr()[0] = 200;
    datast.storage().get_cpu_ptr()[1] = 400;
    EXPECT_EQ(datast.storage().get_cpu_ptr()[0], 200);
    EXPECT_EQ(datast.storage().get_cpu_ptr()[1], 400);
    EXPECT_EQ(datast_cpy.storage().get_cpu_ptr()[0], 200);
    EXPECT_EQ(datast_cpy.storage().get_cpu_ptr()[1], 400);

    data_store<cuda_storage<double>, storage_info_t> ds_cpy_ass1(si);
    data_store<cuda_storage<double>, storage_info_t> ds_cpy_ass2 = ds_cpy_ass1;
    EXPECT_TRUE(ds_cpy_ass2.storage().get_cpu_ptr() == ds_cpy_ass1.storage().get_cpu_ptr());
    EXPECT_TRUE(ds_cpy_ass2.storage().get_target_ptr() == ds_cpy_ass1.storage().get_target_ptr());
    EXPECT_TRUE(ds_cpy_ass2.info() == ds_cpy_ass1.info());
}

TEST(DataStoreTest, States) {
    using data_store_t = data_store<cuda_storage<double>, storage_info_t>;
    storage_info_t si(3, 3, 3);
    // create and allocate data_store
    data_store_t ds(si);

    // intial state should be
    EXPECT_FALSE(ds.storage().state().host_needs_update());
    EXPECT_FALSE(ds.storage().state().device_needs_update());

    // host write views should be valid, this means the device needs an update
    ds.reactivate_host_write_views();
    EXPECT_FALSE(ds.storage().state().host_needs_update());
    EXPECT_TRUE(ds.storage().state().device_needs_update());

    // synchronize everything, valid on both sides
    ds.sync();
    EXPECT_FALSE(ds.storage().state().host_needs_update());
    EXPECT_FALSE(ds.storage().state().device_needs_update());

    // device write views should be valid, this means the host needs an update
    ds.reactivate_target_write_views();
    EXPECT_TRUE(ds.storage().state().host_needs_update());
    EXPECT_FALSE(ds.storage().state().device_needs_update());

    // go back from device
    ds.sync();
    EXPECT_FALSE(ds.storage().state().host_needs_update());
    EXPECT_FALSE(ds.storage().state().device_needs_update());
}

TEST(DataStoreTest, Initializer) {
    storage_info_t si(12, 12, 8);
    data_store<cuda_storage<double>, storage_info_t> ds(si, 3.1415);
    check_vals<<<1, 1>>>(ds.storage().get_target_ptr(), get_gpu_storage_info_ptr(ds.info()));
    ds.clone_from_device();
    for (uint_t i = 0; i < 12; ++i)
        for (uint_t j = 0; j < 12; ++j)
            for (uint_t k = 0; k < 8; ++k)
                EXPECT_EQ(ds.storage().get_cpu_ptr()[si.index(i, j, k)], 1.0);
}

TEST(DataStoreTest, LambdaInitializer) {
    storage_info_t si(10, 11, 12);
    data_store<cuda_storage<double>, storage_info_t> ds(si, [](int i, int j, int k) { return i + j + k; });
    check_vals_lambda<<<1, 1>>>(ds.storage().get_target_ptr(), get_gpu_storage_info_ptr(ds.info()));
    ds.clone_from_device();
    for (uint_t i = 0; i < 10; ++i)
        for (uint_t j = 0; j < 11; ++j)
            for (uint_t k = 0; k < 12; ++k)
                EXPECT_EQ(ds.storage().get_cpu_ptr()[si.index(i, j, k)], (i + j + k + 1.0));
}

TEST(DataStoreTest, Naming) {
    storage_info_t si(10, 11, 12);
    // no naming
    data_store<cuda_storage<double>, storage_info_t> ds2_nn(si);
    data_store<cuda_storage<double>, storage_info_t> ds3_nn(si, 1.0);
    data_store<cuda_storage<double>, storage_info_t> ds4_nn(si, [](int i, int j, int k) { return i + j + k; });
    EXPECT_EQ(ds2_nn.name(), "");
    EXPECT_EQ(ds3_nn.name(), "");
    EXPECT_EQ(ds4_nn.name(), "");

    // test naming
    data_store<cuda_storage<double>, storage_info_t> ds2(si, "standard storage");
    data_store<cuda_storage<double>, storage_info_t> ds3(si, 1.0, "value init. storage");
    data_store<cuda_storage<double>, storage_info_t> ds4(
        si, [](int i, int j, int k) { return i + j + k; }, "lambda init. storage");
    EXPECT_EQ(ds2.name(), "standard storage");
    EXPECT_EQ(ds3.name(), "value init. storage");
    EXPECT_EQ(ds4.name(), "lambda init. storage");

    // create a copy and see if still ok
    auto ds2_tmp = ds2;
    EXPECT_EQ(ds2_tmp.name(), "standard storage");
    EXPECT_EQ(ds3.name(), "value init. storage");
}

TEST(DataStoreTest, ExternalPointer) {
    // test with an external CPU pointer
    storage_info_t si(10, 10, 10);
    double *external_ptr = new double[si.padded_total_length()];
    // create a data_store with externally managed storage
    data_store<cuda_storage<double>, storage_info_t> ds(si, external_ptr, ownership::external_cpu);
    ds.sync();
    // create a copy (double free checks)
    data_store<cuda_storage<double>, storage_info_t> ds_cpy = ds;
    // check values
    for (uint_t i = 0; i < 10; ++i)
        for (uint_t j = 0; j < 10; ++j)
            for (uint_t k = 0; k < 10; ++k) {
                external_ptr[si.index(i, j, k)] = 3.1415;
                EXPECT_EQ(ds.storage().get_cpu_ptr()[si.index(i, j, k)], 3.1415);
                EXPECT_EQ(ds_cpy.storage().get_cpu_ptr()[si.index(i, j, k)], 3.1415);
            }
    // delete the ptr
    delete[] external_ptr;
}

TEST(DataStoreTest, DimAndSizeInterface) {
    storage_info_t si(128, 128, 80);
    data_store<cuda_storage<double>, storage_info_t> ds(si, 3.1415);
    EXPECT_EQ(ds.padded_total_length(), si.padded_total_length());
    EXPECT_EQ(ds.total_length<0>(), si.total_length<0>());
    EXPECT_EQ(ds.total_length<1>(), si.total_length<1>());
    EXPECT_EQ(ds.total_length<2>(), si.total_length<2>());
}

TEST(DataStoreTest, ExternalGPUPointer) {
    // test with an external GPU pointer
    storage_info_t si(10, 10, 10);
    double *external_gpu_ptr;
    double *external_cpu_ptr = new double[si.padded_total_length()];
    // initialize CPU ptr
    for (uint_t i = 0; i < si.padded_total_length(); ++i) {
        external_cpu_ptr[i] = 3.1415;
    }
    // create a GPU ptr
    GT_CUDA_CHECK(cudaMalloc(&external_gpu_ptr, si.padded_total_length() * sizeof(double)));
    // initialize the GPU ptr
    GT_CUDA_CHECK(cudaMemcpy(
        external_gpu_ptr, external_cpu_ptr, si.padded_total_length() * sizeof(double), cudaMemcpyHostToDevice));
    // create a data_store with externally managed storage
    data_store<cuda_storage<double>, storage_info_t> ds(si, external_gpu_ptr, ownership::external_gpu);
    ds.sync();
    // create some copies
    data_store<cuda_storage<double>, storage_info_t> ds_cpy_1(ds);
    data_store<cuda_storage<double>, storage_info_t> ds_cpy_2 = ds_cpy_1;
    EXPECT_EQ(ds_cpy_1.storage().get_cpu_ptr(), ds_cpy_2.storage().get_cpu_ptr());
    EXPECT_EQ(ds_cpy_2.storage().get_cpu_ptr(), ds.storage().get_cpu_ptr());
    // create a copy (double free checks)
    data_store<cuda_storage<double>, storage_info_t> ds_cpy = ds;
    // check values
    for (uint_t i = 0; i < 10; ++i)
        for (uint_t j = 0; j < 10; ++j)
            for (uint_t k = 0; k < 10; ++k) {
                EXPECT_EQ(ds.storage().get_cpu_ptr()[si.index(i, j, k)], 3.1415);
                EXPECT_EQ(ds_cpy.storage().get_cpu_ptr()[si.index(i, j, k)], 3.1415);
            }
    // delete the ptr
    delete[] external_cpu_ptr;
    GT_CUDA_CHECK(cudaFree(external_gpu_ptr));
}
