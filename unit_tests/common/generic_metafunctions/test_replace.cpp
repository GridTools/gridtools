/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <boost/mpl/vector/vector10.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/generic_metafunctions/replace.hpp>

using namespace gridtools;

TEST(test_replace, test) {
    GT_STATIC_ASSERT(
        (boost::mpl::equal<replace<boost::mpl::vector4<int, double, char, long>, static_uint<3>, float>::type,
            boost::mpl::vector4<int, double, char, float>>::value),
        "Error");

    GT_STATIC_ASSERT(
        (boost::mpl::equal<replace<boost::mpl::vector4<int, double, char, long>, static_uint<2>, float>::type,
            boost::mpl::vector4<int, double, float, long>>::value),
        "Error");

    ASSERT_TRUE(true);
}
