/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "gtest/gtest.h"

#include <gridtools/common/gt_assert.hpp>
#include <gridtools/storage/storage_host/host_storage.hpp>

TEST(StorageHostTest, Simple) {
    // create two storages
    gridtools::host_storage<int> s1(2);
    gridtools::host_storage<int> s2(2);
    // test the is_storage check
    GT_STATIC_ASSERT(gridtools::is_storage<decltype(s1)>::type::value, "is_storage check is not working anymore");
    GT_STATIC_ASSERT(!gridtools::is_storage<int>::type::value, "is_storage check is not working anymore");
    // write some values
    s1.get_cpu_ptr()[0] = 10;
    s1.get_cpu_ptr()[1] = 20;
    s2.get_cpu_ptr()[0] = 100;
    s2.get_cpu_ptr()[1] = 200;
    // check if they are there
    EXPECT_EQ(s1.get_cpu_ptr()[1], 20);
    EXPECT_EQ(s1.get_cpu_ptr()[0], 10);
    EXPECT_EQ(s2.get_cpu_ptr()[1], 200);
    EXPECT_EQ(s2.get_cpu_ptr()[0], 100);
    // ptr ref should be equal to the cpu ptr
    EXPECT_EQ(s1.get_cpu_ptr(), s1.get_ptrs<typename gridtools::host_storage<int>::ptrs_t>());
    EXPECT_EQ(s2.get_cpu_ptr(), s2.get_ptrs<typename gridtools::host_storage<int>::ptrs_t>());
    // swap storages
    s1.swap(s2);
    // check if changes are there
    EXPECT_EQ(s2.get_cpu_ptr()[1], 20);
    EXPECT_EQ(s2.get_cpu_ptr()[0], 10);
    EXPECT_EQ(s1.get_cpu_ptr()[1], 200);
    EXPECT_EQ(s1.get_cpu_ptr()[0], 100);
}

TEST(StorageHostTest, InitializedStorage) {
    // create two storages
    gridtools::host_storage<int> s1(2, [](int) { return 3; });
    gridtools::host_storage<int> s2(2, [](int) { return 5; });
    // check values
    EXPECT_EQ(s1.get_cpu_ptr()[0], 3);
    EXPECT_EQ(s1.get_cpu_ptr()[1], 3);
    EXPECT_EQ(s2.get_cpu_ptr()[0], 5);
    EXPECT_EQ(s2.get_cpu_ptr()[1], 5);
    // change one value
    s1.get_cpu_ptr()[0] = 10;
    s2.get_cpu_ptr()[1] = 20;
    // check values
    EXPECT_EQ(s1.get_cpu_ptr()[0], 10);
    EXPECT_EQ(s1.get_cpu_ptr()[1], 3);
    EXPECT_EQ(s2.get_cpu_ptr()[0], 5);
    EXPECT_EQ(s2.get_cpu_ptr()[1], 20);
}
