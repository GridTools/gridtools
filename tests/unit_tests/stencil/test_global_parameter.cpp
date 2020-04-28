/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/stencil/global_parameter.hpp>

#include <gtest/gtest.h>

#include <gridtools/sid/concept.hpp>

namespace gridtools {
    namespace stencil {
        namespace {
            TEST(global_parameter, smoke) {
                auto testee = make_global_parameter(42.);
                using testee_t = decltype(testee);

                static_assert(is_sid<testee_t>(), "");
                static_assert(std::is_same<sid::element_type<testee_t>, double>(), "");
                static_assert(std::is_same<sid::reference_type<testee_t>, double>(), "");
                static_assert(sizeof(sid::ptr_holder_type<testee_t>) == sizeof(double), "");
                static_assert(sizeof(sid::ptr_type<testee_t>) == sizeof(double), "");
                static_assert(std::is_empty<sid::ptr_diff_type<testee_t>>(), "");
                static_assert(std::is_empty<sid::strides_type<testee_t>>(), "");
                static_assert(tuple_util::size<sid::strides_type<testee_t>>::value == 0, "");

                EXPECT_EQ(42., *sid::get_origin(testee)());
            }
        } // namespace
    }     // namespace stencil
} // namespace gridtools
