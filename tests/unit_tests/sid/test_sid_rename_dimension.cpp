/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/sid/rename_dimension.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/hymap.hpp>
#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/simple_ptr_holder.hpp>
#include <gridtools/sid/synthetic.hpp>

namespace gridtools {
    namespace {
        using sid::property;
        using namespace literals;
        namespace tu = tuple_util;

        struct a {};
        struct b {};
        struct c {};
        struct d {};

        TEST(rename_dimensions, smoke) {
            double data[3][5][7];

            auto src = sid::synthetic()
                           .set<property::origin>(sid::make_simple_ptr_holder(&data[0][0][0]))
                           .set<property::strides>(tu::make<hymap::keys<a, b, c>::values>(5_c * 7_c, 7_c, 1_c))
                           .set<property::upper_bounds>(tu::make<hymap::keys<a, b>::values>(3, 5));

            auto testee = sid::rename_dimension<b, d>(src);
            using testee_t = decltype(testee);

            auto strides = sid::get_strides(testee);
            EXPECT_EQ(35, sid::get_stride<a>(strides));
            EXPECT_EQ(0, sid::get_stride<b>(strides));
            EXPECT_EQ(1, sid::get_stride<c>(strides));
            EXPECT_EQ(7, sid::get_stride<d>(strides));

            static_assert(meta::is_empty<get_keys<sid::lower_bounds_type<testee_t>>>(), "");

            auto u_bound = sid::get_upper_bounds(testee);
            EXPECT_EQ(3, at_key<a>(u_bound));
            EXPECT_EQ(5, at_key<d>(u_bound));
        }

        TEST(rename_dimensions, rename_twice_and_make_composite) {
            double data[3][5][7];
            auto src = sid::synthetic()
                           .set<property::origin>(sid::make_simple_ptr_holder(&data[0][0][0]))
                           .set<property::strides>(tu::make<hymap::keys<a, b, c>::values>(5_c * 7_c, 7_c, 1_c))
                           .set<property::upper_bounds>(tu::make<hymap::keys<a, b>::values>(3, 5));
            auto testee = sid::rename_dimension<a, c>(sid::rename_dimension<b, d>(src));
            static_assert(sid::is_sid<decltype(testee)>(), "");
            auto composite = tu::make<gridtools::sid::composite::keys<void>::values>(testee);
            static_assert(sid::is_sid<decltype(composite)>(), "");
        }
    } // namespace
} // namespace gridtools
