/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/generic_metafunctions/shorten.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>

using namespace gridtools;

template <int_t... Vars>
struct aex;

TEST(shorten, 4) {
    GT_STATIC_ASSERT((std::is_same<shorten<int_t, aex<3, 4, 5, 6, 8, 9>, 3>::type, aex<3, 4, 5>>::value), "ERROR");

    GT_STATIC_ASSERT((std::is_same<shorten<int_t, aex<3, 4, 5, 6, 8, 9>, 1>::type, aex<3>>::value), "ERROR");

    GT_STATIC_ASSERT(
        (std::is_same<shorten<int_t, aex<3, 4, 5, 6, 8, 9>, 6>::type, aex<3, 4, 5, 6, 8, 9>>::value), "ERROR");
}
