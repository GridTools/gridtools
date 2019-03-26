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
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <iostream>
#include <utility>

namespace gt = gridtools;

TEST(Storage, Swap) {
    using storage_info_t = gt::storage_traits<backend_t::backend_target_t>::storage_info_t<0, 3>;
    using data_store_t = gt::storage_traits<backend_t::backend_target_t>::data_store_t<double, storage_info_t>;

    storage_info_t s1(3, 3, 3);
    data_store_t ds1(s1, "ds1");

    storage_info_t s2(3, 30, 20);
    data_store_t ds2(s2, "ds2");

    auto name1 = ds1.name();
    auto ptr1 = ds1.get_storage_ptr();
    auto iptr1 = ds1.get_storage_info_ptr();

    auto name2 = ds2.name();
    auto ptr2 = ds2.get_storage_ptr();
    auto iptr2 = ds2.get_storage_info_ptr();

    using std::swap;
    swap(ds1, ds2);

    EXPECT_EQ(name1, ds2.name());
    EXPECT_EQ(ptr1, ds2.get_storage_ptr());
    EXPECT_EQ(iptr1, ds2.get_storage_info_ptr());

    EXPECT_EQ(name2, ds1.name());
    EXPECT_EQ(ptr2, ds1.get_storage_ptr());
    EXPECT_EQ(iptr2, ds1.get_storage_info_ptr());
}
