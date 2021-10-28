/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/str_literal.hpp>

#include <cstring>
#include <string_view>
#include <type_traits>

#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        template <str_literal>
        struct foo {};

        static_assert(std::is_same_v<foo<"abcde">, foo<"abcde">>);
        static_assert(!std::is_same_v<foo<"abcde">, foo<"qwerty">>);

        static_assert(str_literal("qwerty") == std::string_view("qwerty"));

        TEST(str_literal, c_string_conversion) { EXPECT_EQ(std::strcmp(str_literal("foo"), "foo"), 0); }

        TEST(str_literal, string_view_conversion) {
            std::string_view view = str_literal("foo");
            EXPECT_EQ(view, "foo");
        }

        TEST(str_literal, string_conversion) {
            std::string string = str_literal("foo");
            EXPECT_EQ(string, "foo");
        }
    } // namespace
} // namespace gridtools