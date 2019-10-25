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
#include <gridtools/storage/storage_host/host_storage.hpp>

TEST(StorageHostTest, Simple) {
    // create two storages
    gridtools::host_storage<int> s1(2, 0, gridtools::alignment<1>());
    gridtools::host_storage<int> s2(2, 0, gridtools::alignment<1>());
    // test the is_storage check
    static_assert(gridtools::is_storage<decltype(s1)>::value, "");
    static_assert(!gridtools::is_storage<int>::value, "");
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
}
