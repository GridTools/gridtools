/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/sid/sid_shift_origin.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/stencil_composition/sid/simple_ptr_holder.hpp>
#include <gridtools/stencil_composition/sid/synthetic.hpp>

namespace gridtools {
    namespace {
        using sid::property;
        using namespace literals;
        namespace tu = tuple_util;

        TEST(shift_sid_origin, synthetic) {
            double data[3][5][7];

            auto src = sid::synthetic()
                           .set<property::origin>(sid::make_simple_ptr_holder(&data[0][0][0]))
                           .set<property::strides>(tu::make<tuple>(5_c * 7_c, 7_c, 1_c))
                           .set<property::upper_bounds>(tu::make<tuple>(3));

            auto offset = tuple_util::make<tuple>(1_c, 2);
            auto testee = sid::shift_sid_origin(src, offset);

            static_assert(is_sid<decltype(testee)>(), "");

            EXPECT_EQ(&data[0][0][0], sid::get_origin(src)());
            EXPECT_EQ(&data[1][2][0], sid::get_origin(testee)());

            EXPECT_EQ(2, tu::get<0>(sid::get_upper_bounds(testee)));
        }

        TEST(shift_sid_origin, c_array) {
            double data[3][5][7];

            auto offset = tuple_util::make<tuple>(1_c, 2);
            auto testee = sid::shift_sid_origin(data, offset);

            static_assert(is_sid<decltype(testee)>(), "");

            EXPECT_EQ(&data[1][2][0], sid::get_origin(testee)());

            auto lower_bounds = sid::get_lower_bounds(testee);
            EXPECT_EQ(-1, tu::get<0>(lower_bounds));
            EXPECT_EQ(-2, tu::get<1>(lower_bounds));
            EXPECT_EQ(0, tu::get<2>(lower_bounds));

            auto upper_bounds = sid::get_upper_bounds(testee);
            EXPECT_EQ(2, tu::get<0>(upper_bounds));
            EXPECT_EQ(3, tu::get<1>(upper_bounds));
            EXPECT_EQ(7, tu::get<2>(upper_bounds));
        }
    } // namespace
} // namespace gridtools
