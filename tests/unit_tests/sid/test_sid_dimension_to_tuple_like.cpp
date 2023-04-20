/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/sid/dimension_to_tuple_like.hpp>

namespace gridtools {
    namespace {
        TEST(dimension_to_tuple_like, smoke) {
            double data[3][4] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
            auto testee = sid::dimension_to_tuple_like<integral_constant<int, 0>, 3>(data);
            static_assert(is_sid<decltype(testee)>::value);

            auto ptr = sid::get_origin(testee)();
            auto strides = sid::get_strides(testee);

            static_assert(tuple_util::size<decltype(strides)>{} == 1);

            EXPECT_EQ(data[1][0], tuple_util::get<1>(*ptr));
            EXPECT_EQ(&data[2][0], &tuple_util::get<2>(*ptr));

            sid::shift(ptr, tuple_util::get<0>(strides), 2);
            EXPECT_EQ(data[1][2], tuple_util::get<1>(*ptr));
            EXPECT_EQ(&data[2][2], &tuple_util::get<2>(*ptr));

            tuple_util::get<1>(*sid::shifted(ptr, tuple_util::get<0>(strides), 1)) = 42;
            EXPECT_EQ(42, data[1][3]);
        }

        TEST(dimension_to_tuple_like, assignable_from_tuple_like) {
            double data[3][4] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
            auto testee = sid::dimension_to_tuple_like<gridtools::integral_constant<int, 0>, 2>(data);
            static_assert(is_sid<decltype(testee)>::value);

            auto ptr = sid::get_origin(testee)();

            *ptr = tuple(2., 3.);
            EXPECT_EQ(tuple_util::get<0>(*ptr), 2.);
            EXPECT_EQ(data[0][0], 2.);
            EXPECT_EQ(tuple_util::get<1>(*ptr), 3.);
            EXPECT_EQ(data[1][0], 3.);
        }

        TEST(dimension_to_tuple_like, nested) {
            double data[2][3] = {0, 1, 2, 3, 4, 5};
            auto testee = sid::dimension_to_tuple_like<integral_constant<int, 1>, 3>(
                sid::dimension_to_tuple_like<integral_constant<int, 0>, 2>(data));
            static_assert(is_sid<decltype(testee)>::value);

            auto derefed_ptr = *sid::get_origin(testee)();

            auto first_level = tuple_util::get<2>(derefed_ptr);
            ASSERT_EQ(data[0][2], tuple_util::get<0>(first_level));
            ASSERT_EQ(data[1][2], tuple_util::get<1>(first_level));

            EXPECT_EQ(&data[1][2], &tuple_util::get<1>(tuple_util::get<2>(derefed_ptr)));

            derefed_ptr = array<array<double, 2>, 3>{10, 11, 12, 13, 14, 15};
            EXPECT_EQ(data[1][2], 15);
        }

    } // namespace
} // namespace gridtools
