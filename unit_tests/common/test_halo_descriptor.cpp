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
#include <gridtools/common/defs.hpp>
#include <gridtools/common/halo_descriptor.hpp>

using namespace gridtools;

TEST(test_halo_descriptor, is_valid) {
    uint_t size = 7;
    uint_t halo_size = 3;

    ASSERT_NO_THROW((halo_descriptor{halo_size, halo_size, halo_size, size - halo_size - 1, size}));
}

TEST(test_halo_descriptor, default_constructed_is_valid) { ASSERT_NO_THROW((halo_descriptor())); }
