/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/variadic_pack_metafunctions.hpp>

#include <gtest/gtest.h>

using namespace gridtools;

static_assert(get_index_of_element_in_pack(0, 1, 1, 2, 3, 4) == 0, "");
static_assert(get_index_of_element_in_pack(0, 2, 1, 2, 3, 4) == 1, "");
static_assert(get_index_of_element_in_pack(0, 3, 1, 2, 3, 4) == 2, "");
static_assert(get_index_of_element_in_pack(0, 4, 1, 2, 3, 4) == 3, "");

static_assert(get_value_from_pack(0, 1, 2, 3, 4) == 1, "");
static_assert(get_value_from_pack(1, 1, 2, 3, 4) == 2, "");
static_assert(get_value_from_pack(2, 1, 2, 3, 4) == 3, "");
static_assert(get_value_from_pack(3, 1, 2, 3, 4) == 4, "");

TEST(dummy, dummy) {}
