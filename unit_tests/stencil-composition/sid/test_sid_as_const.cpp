/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil-composition/sid/as_const.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/meta/macros.hpp>
#include <gridtools/stencil-composition/sid/concept.hpp>
#include <gridtools/stencil-composition/sid/synthetic.hpp>

namespace gridtools {
    namespace {
        using sid::property;

        TEST(as_const, smoke) {
            double data = 42;
            auto src = sid::synthetic().set<property::origin>(&data);
            auto testee = sid::as_const(src);
            using testee_t = decltype(testee);

            static_assert(is_sid<testee_t>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_type, testee_t), double const *>(), "");
            EXPECT_EQ(sid::get_origin(src), sid::get_origin(testee));
        }
    } // namespace
} // namespace gridtools
