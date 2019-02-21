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

#include <tuple>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/generic_metafunctions/unzip.hpp>
#include <gridtools/common/gt_assert.hpp>

using namespace gridtools;

TEST(unzip, do_unzip) {

    typedef std::tuple<int, float, double, char, bool, short> list_t;
    // verifying that the information is actually compile-time known and that it's correct
    GT_STATIC_ASSERT(
        (std::is_same<std::tuple<int, double, bool>, typename unzip<list_t>::first>::value), "error on first argument");
    GT_STATIC_ASSERT((std::is_same<std::tuple<float, char, short>, typename unzip<list_t>::second>::value),
        "error on second argument");
}
