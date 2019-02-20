/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil-composition/sid/synthetic.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/array.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/meta/macros.hpp>
#include <gridtools/stencil-composition/sid/concept.hpp>

namespace gridtools {
    namespace {

        using sid::property;

        TEST(sid_synthetic, smoke) {
            double a = 100;
            auto testee = sid::synthetic().set<property::origin>(&a);
            static_assert(is_sid<decltype(testee)>::value, "");

            EXPECT_EQ(a, *sid::get_origin(testee));
        }

        namespace custom {
            struct element {};
            struct ptr_diff {
                int val;
            };
            struct ptr {
                element const *val;
                GT_FUNCTION element const &operator*() const { return *val; }
                friend GT_FUNCTION ptr operator+(ptr, ptr_diff) { return {}; }
            };
            struct stride {
                int val;
                friend GT_FUNCTION std::true_type sid_shift(ptr &, stride const &, int) { return {}; }
                friend GT_FUNCTION std::false_type sid_shift(ptr_diff &, stride const &, int) { return {}; }
            };
            using strides = array<stride, 2>;

            struct strides_kind;

            TEST(sid_synthetic, custom) {
                element the_element = {};
                ptr the_origin = {&the_element};
                strides the_strides = {stride{3}, stride{4}};

                auto the_testee = sid::synthetic()
                                      .set<property::origin>(the_origin)
                                      .set<property::strides>(the_strides)
                                      .set<property::ptr_diff, ptr_diff>()
                                      .set<property::strides_kind, strides_kind>();

                using testee = decltype(the_testee);

                static_assert(is_sid<testee>(), "");
                static_assert(std::is_trivially_copyable<testee>(), "");

                static_assert(std::is_same<GT_META_CALL(sid::ptr_type, testee), ptr>(), "");
                static_assert(std::is_same<GT_META_CALL(sid::strides_type, testee), strides>(), "");
                static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, testee), ptr_diff>(), "");
                static_assert(std::is_same<GT_META_CALL(sid::strides_kind, testee), strides_kind>(), "");

                EXPECT_EQ(&the_element, sid::get_origin(the_testee).val);
                EXPECT_EQ(3, tuple_util::get<0>(sid::get_strides(the_testee)).val);
                EXPECT_EQ(4, tuple_util::get<1>(sid::get_strides(the_testee)).val);
            }
        } // namespace custom
    }     // namespace
} // namespace gridtools
