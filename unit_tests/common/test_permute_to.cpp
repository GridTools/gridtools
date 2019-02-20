/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/permute_to.hpp>

#include <utility>

#include <gtest/gtest.h>

namespace gridtools {

    TEST(permute_to, lref) {
        std::tuple<> src;
        EXPECT_TRUE(permute_to<std::tuple<>>(src) == std::make_tuple());
    }

    TEST(permute_to, cref) {
        std::tuple<> const src = {};
        EXPECT_TRUE(permute_to<std::tuple<>>(src) == std::make_tuple());
    }

    template <typename Res, typename... Args>
    Res testee(Args &&... args) {
        return permute_to<Res>(std::make_tuple(std::forward<Args>(args)...));
    }

    TEST(permute_to, empty) { EXPECT_TRUE(testee<std::tuple<>>() == std::make_tuple()); }

    TEST(permute_to, one) { EXPECT_TRUE(testee<std::tuple<int>>(42) == std::make_tuple(42)); }

    TEST(permute_to, functional) {
        using res_t = std::tuple<int, char, double>;
        res_t expected{42, 'a', .1};
        EXPECT_TRUE(testee<res_t>(42, 'a', .1) == expected);
        EXPECT_TRUE(testee<res_t>(42, .1, 'a') == expected);
        EXPECT_TRUE(testee<res_t>('a', 42, .1) == expected);
        EXPECT_TRUE(testee<res_t>('a', .1, 42) == expected);
        EXPECT_TRUE(testee<res_t>(.1, 42, 'a') == expected);
        EXPECT_TRUE(testee<res_t>(.1, 'a', 42) == expected);
    }

    TEST(permute_to, unused_extra_args) {
        EXPECT_TRUE((testee<std::tuple<int>>('a', 42, .1, true) == std::make_tuple(42)));
    }

    TEST(permute_to, duplicates_in_res) { EXPECT_TRUE((testee<std::tuple<int, int>>(42) == std::make_tuple(42, 42))); }
} // namespace gridtools
