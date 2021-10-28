/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <algorithm>
#include <string>
#include <string_view>

#if __cplusplus < 202002L
#error str_literal is only usable in C++20
#endif

namespace gridtools {
    /**
     * Usage:
     *   str_literal can be used to define template non-type parameter:
     *
     *   template <str_literal Name> struct foo {};
     *
     *   Now that template can be instantiated with a string literal:
     *
     *   foo<"abcd">
     */
    template <size_t N>
    struct str_literal {
        char value[N];
        constexpr str_literal(const char (&str)[N]) { std::ranges::copy(str, value); }
        constexpr operator char const *() const { return value; }
        constexpr operator std::string_view() const { return value; }
        operator std::string() const { return value; }
    };
} // namespace gridtools