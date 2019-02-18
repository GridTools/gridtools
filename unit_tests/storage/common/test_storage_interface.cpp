/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "gtest/gtest.h"

#include <gridtools/storage/common/storage_interface.hpp>

using namespace gridtools;

// storage that implements the storage_interface
struct host_storage : storage_interface<host_storage> {
    int a;
    host_storage(uint_t i) : a(i) {}
    void clone_to_device_impl() { a *= 2; }
};

// another storage that implements the storage_interface
struct cuda_storage : storage_interface<cuda_storage> {
    int a;
    cuda_storage(uint_t i) : a(i) {}
    void clone_to_device_impl() { a *= 3; }
};

TEST(StorageInterface, Simple) {
    host_storage h(10);
    cuda_storage c(10);

    EXPECT_EQ(h.a, 10);
    EXPECT_EQ(c.a, 10);

    h.clone_to_device();
    c.clone_to_device();

    EXPECT_EQ(h.a, 20);
    EXPECT_EQ(c.a, 30);
}

TEST(StorageInterface, Sizes) {
    cuda_storage c(10);
    host_storage h(10);
    // the sizes should stay 1, because the idea is that the
    // storage interface is only providing a set of methods
    EXPECT_EQ(sizeof(storage_interface<host_storage>), 1);
    EXPECT_EQ(sizeof(storage_interface<cuda_storage>), 1);
}
