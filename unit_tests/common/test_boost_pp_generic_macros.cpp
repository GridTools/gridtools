/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/boost_pp_generic_macros.hpp>
#include <gtest/gtest.h>

#define my_types ((int))((double))
GT_PP_MAKE_VARIANT(myvariant, my_types);
#undef my_types
TEST(variant, automatic_conversion) {
    myvariant v = 3;
    (int)v;

    v = 3.;
    (double)v;

    try {
        (int)v;
        ASSERT_TRUE(false);
    } catch (const boost::bad_get &e) {
        ASSERT_TRUE(true);
    }
}

#define my_types ((int, 3))((double, 1))
GT_PP_MAKE_VARIANT(myvariant_tuple, my_types);
#undef my_types
TEST(variant_with_tuple, automatic_conversion) {
    myvariant_tuple v = 3;
    (int)v;

    v = 3.;
    (double)v;

    try {
        (int)v;
        ASSERT_TRUE(false);
    } catch (const boost::bad_get &e) {
        ASSERT_TRUE(true);
    }
}
