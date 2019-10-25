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
#include <gridtools/storage/common/storage_info.hpp>
#include <gridtools/storage/data_store.hpp>
#include <gridtools/storage/storage_host/host_storage.hpp>

using namespace gridtools;

typedef storage_info<0, layout_map<0, 1, 2>> storage_info_t;
typedef storage_info<0, layout_map<0, 1, 2>, halo<2, 1, 0>> storage_info_halo_t;
typedef storage_info<0, layout_map<0, 1, 2>, halo<2, 1, 0>, alignment<16>> storage_info_halo_aligned_t;

TEST(DataStoreTest, Simple) {
    storage_info_t si(3, 3, 3);
    storage_info_halo_t si_halo(7, 5, 3);
    storage_info_halo_aligned_t si_halo_al(7, 5, 3);
    // check sizes, strides, and alignment
    ASSERT_EQ(si.padded_length<0>(), 3);
    ASSERT_EQ(si.padded_length<1>(), 3);
    ASSERT_EQ(si.padded_length<2>(), 3);
    ASSERT_EQ(si.total_length<0>(), 3);
    ASSERT_EQ(si.total_length<1>(), 3);
    ASSERT_EQ(si.total_length<2>(), 3);
    ASSERT_EQ(si.length<0>(), 3);
    ASSERT_EQ(si.length<1>(), 3);
    ASSERT_EQ(si.length<2>(), 3);
    ASSERT_EQ(si.stride<0>(), 9);
    ASSERT_EQ(si.stride<1>(), 3);
    ASSERT_EQ(si.stride<2>(), 1);

    ASSERT_EQ(si_halo.template padded_length<0>(), 7);
    ASSERT_EQ(si_halo.template padded_length<1>(), 5);
    ASSERT_EQ(si_halo.template padded_length<2>(), 3);
    ASSERT_EQ(si_halo.template total_length<0>(), 7);
    ASSERT_EQ(si_halo.template total_length<1>(), 5);
    ASSERT_EQ(si_halo.template total_length<2>(), 3);
    ASSERT_EQ(si_halo.template length<0>(), 3);
    ASSERT_EQ(si_halo.template length<1>(), 3);
    ASSERT_EQ(si_halo.template length<2>(), 3);
    ASSERT_EQ(si_halo.template stride<0>(), 15);
    ASSERT_EQ(si_halo.template stride<1>(), 3);
    ASSERT_EQ(si_halo.template stride<2>(), 1);

    ASSERT_EQ(si_halo_al.template padded_length<0>(), 7);
    ASSERT_EQ(si_halo_al.template padded_length<1>(), 5);
    ASSERT_EQ(si_halo_al.template padded_length<2>(), 16);
    ASSERT_EQ(si_halo_al.template total_length<0>(), 7);
    ASSERT_EQ(si_halo.template total_length<1>(), 5);
    ASSERT_EQ(si_halo.template total_length<2>(), 3);
    ASSERT_EQ(si_halo.template length<0>(), 3);
    ASSERT_EQ(si_halo.template length<1>(), 3);
    ASSERT_EQ(si_halo.template length<2>(), 3);
    ASSERT_EQ(si_halo_al.template stride<0>(), 80);
    ASSERT_EQ(si_halo_al.template stride<1>(), 16);
    ASSERT_EQ(si_halo_al.template stride<2>(), 1);

    // create a copy of a data_store and check equivalence
    data_store<host_storage<double>, storage_info_t> datast(si, 5.3);
    data_store<host_storage<double>, storage_info_t> datast_cpy(datast);
    EXPECT_EQ(&datast.info(), &datast_cpy.info());
    EXPECT_EQ(&datast.storage(), &datast_cpy.storage());
    // modify the data and check if the copy can see the changes
    EXPECT_EQ(datast.storage().get_cpu_ptr()[0], 5.3);
    EXPECT_EQ(datast.storage().get_cpu_ptr()[1], 5.3);

    datast.storage().get_cpu_ptr()[0] = 100;
    datast.storage().get_cpu_ptr()[1] = 200;

    EXPECT_EQ(datast.storage().get_cpu_ptr()[0], 100);
    EXPECT_EQ(datast.storage().get_cpu_ptr()[1], 200);

    EXPECT_EQ(datast_cpy.storage().get_cpu_ptr()[0], 100);
    EXPECT_EQ(datast_cpy.storage().get_cpu_ptr()[1], 200);

    // test some copy ctor operations
    data_store<host_storage<double>, storage_info_t> ds_cpy_ass1(si);
    data_store<host_storage<double>, storage_info_t> ds_cpy_ass2 = ds_cpy_ass1;
    EXPECT_EQ(ds_cpy_ass2.storage().get_cpu_ptr(), ds_cpy_ass1.storage().get_cpu_ptr());
    EXPECT_TRUE(ds_cpy_ass2.info() == ds_cpy_ass1.info());
}

TEST(DataStoreTest, Initializer) {
    storage_info_t si(128, 128, 80);
    data_store<host_storage<double>, storage_info_t> ds(si, 3.1415);
    for (uint_t i = 0; i < 128; ++i)
        for (uint_t j = 0; j < 128; ++j)
            for (uint_t k = 0; k < 80; ++k)
                EXPECT_EQ(ds.storage().get_cpu_ptr()[si.index(i, j, k)], 3.1415);
}

TEST(DataStoreTest, LambdaInitializer) {
    storage_info_t si(10, 11, 12);
    data_store<host_storage<double>, storage_info_t> ds(si, [](int i, int j, int k) { return i + j + k; });
    for (uint_t i = 0; i < 10; ++i)
        for (uint_t j = 0; j < 11; ++j)
            for (uint_t k = 0; k < 12; ++k)
                EXPECT_EQ(ds.storage().get_cpu_ptr()[si.index(i, j, k)], (i + j + k));
}

TEST(DataStoreTest, Naming) {
    storage_info_t si(10, 11, 12);
    // no naming
    data_store<host_storage<double>, storage_info_t> ds2_nn(si);
    data_store<host_storage<double>, storage_info_t> ds3_nn(si, 1.0);
    data_store<host_storage<double>, storage_info_t> ds4_nn(si, [](int i, int j, int k) { return i + j + k; });
    EXPECT_EQ(ds2_nn.name(), "");
    EXPECT_EQ(ds3_nn.name(), "");
    EXPECT_EQ(ds4_nn.name(), "");

    // test naming
    data_store<host_storage<double>, storage_info_t> ds2(si, "standard storage");
    data_store<host_storage<double>, storage_info_t> ds3(si, 1.0, "value init. storage");
    data_store<host_storage<double>, storage_info_t> ds4(
        si, [](int i, int j, int k) { return i + j + k; }, "lambda init. storage");
    EXPECT_EQ(ds2.name(), "standard storage");
    EXPECT_EQ(ds3.name(), "value init. storage");
    EXPECT_EQ(ds4.name(), "lambda init. storage");

    // create a copy and see if still ok
    auto ds2_tmp = ds2;
    EXPECT_EQ(ds2_tmp.name(), "standard storage");
    auto ds1 = ds3;
    EXPECT_EQ(ds1.name(), "value init. storage");
    EXPECT_EQ(ds3.name(), "value init. storage");
}

TEST(DataStoreTest, InvalidSize) {
    EXPECT_THROW(storage_info_t(128, 128, 0), std::runtime_error);
    EXPECT_THROW(storage_info_t(-128, 128, 80), std::runtime_error);
}

TEST(DataStoreTest, DimAndSizeInterface) {
    storage_info_t si(128, 128, 80);
    data_store<host_storage<double>, storage_info_t> ds(si, 3.1415);
    EXPECT_EQ(ds.padded_total_length(), si.padded_total_length());
    EXPECT_EQ(ds.total_length<0>(), si.total_length<0>());
    EXPECT_EQ(ds.total_length<1>(), si.total_length<1>());
    EXPECT_EQ(ds.total_length<2>(), si.total_length<2>());
}

TEST(DataStoreTest, ExternalPointer) {
    double *external_ptr = new double[10 * 10 * 10];
    storage_info_t si(10, 10, 10);
    // create a data_store with externally managed storage
    data_store<host_storage<double>, storage_info_t> ds(si, external_ptr);
    // create some copies
    data_store<host_storage<double>, storage_info_t> ds_cpy_1(ds);
    data_store<host_storage<double>, storage_info_t> ds_cpy_2 = ds_cpy_1;
    EXPECT_EQ(ds_cpy_1.storage().get_cpu_ptr(), ds_cpy_2.storage().get_cpu_ptr());
    EXPECT_EQ(ds_cpy_2.storage().get_cpu_ptr(), ds.storage().get_cpu_ptr());
    // create a copy (double free checks)
    data_store<host_storage<double>, storage_info_t> ds_cpy = ds;
    // check values
    int z = 0;
    for (uint_t i = 0; i < 10; ++i)
        for (uint_t j = 0; j < 10; ++j)
            for (uint_t k = 0; k < 10; ++k) {
                external_ptr[z] = 3.1415;
                z++;
                EXPECT_EQ(ds.storage().get_cpu_ptr()[si.index(i, j, k)], 3.1415);
                EXPECT_EQ(ds_cpy.storage().get_cpu_ptr()[si.index(i, j, k)], 3.1415);
            }
    // delete the ptr
    delete[] external_ptr;
}
