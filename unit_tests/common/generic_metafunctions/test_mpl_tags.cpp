/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <boost/mpl/arithmetic.hpp>
#include <boost/mpl/comparison.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/generic_metafunctions/mpl_tags.hpp>
#include <gtest/gtest.h>

TEST(integralconstant, comparison) {
    GT_STATIC_ASSERT(
        (boost::mpl::greater<std::integral_constant<int, 5>, std::integral_constant<int, 4>>::type::value), "");

    GT_STATIC_ASSERT(
        (boost::mpl::less<std::integral_constant<int, 4>, std::integral_constant<int, 5>>::type::value), "");

    GT_STATIC_ASSERT(
        (boost::mpl::greater_equal<std::integral_constant<int, 5>, std::integral_constant<int, 4>>::type::value), "");

    GT_STATIC_ASSERT(
        (boost::mpl::less_equal<std::integral_constant<int, 4>, std::integral_constant<int, 5>>::type::value), "");
}

TEST(integralconstant, arithmetic) {
    GT_STATIC_ASSERT(
        (boost::mpl::plus<std::integral_constant<int, 5>, std::integral_constant<int, 4>>::type::value == 9), "");
    GT_STATIC_ASSERT(
        (boost::mpl::minus<std::integral_constant<int, 5>, std::integral_constant<int, 4>>::type::value == 1), "");
}
