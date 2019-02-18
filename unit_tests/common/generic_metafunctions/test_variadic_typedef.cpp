/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/generic_metafunctions/variadic_typedef.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>

using namespace gridtools;

TEST(variadic_typedef, test) {

    typedef variadic_typedef<int, double, unsigned int> tt;

    GT_STATIC_ASSERT((std::is_same<tt::template get_elem<0>::type, int>::value), "Error");

    GT_STATIC_ASSERT((std::is_same<tt::template get_elem<1>::type, double>::value), "Error");

    GT_STATIC_ASSERT((std::is_same<tt::template get_elem<2>::type, unsigned int>::value), "Error");
}

TEST(variadic_typedef, get_from_variadic_pack) {

    GT_STATIC_ASSERT((static_int<get_from_variadic_pack<3>::apply(2, 6, 8, 3, 5)>::value == 3), "Error");

    GT_STATIC_ASSERT(
        (static_int<get_from_variadic_pack<7>::apply(2, 6, 8, 3, 5, 4, 6, -8, 4, 3, 1, 54, 67)>::value == -8), "Error");
}

TEST(variadic_typedef, find) {

    typedef variadic_typedef<int, double, unsigned int, double> tt;

    GT_STATIC_ASSERT((tt::find<int>() == 0), "ERROR");
    GT_STATIC_ASSERT((tt::find<double>() == 1), "ERROR");
    GT_STATIC_ASSERT((tt::find<unsigned int>() == 2), "ERROR");
}
