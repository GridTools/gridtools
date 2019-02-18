/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include <type_traits>

/**
 * Compare 2 types for equality. Will produce a readable error message (hopefully).
 */
template <typename expected, typename actual__, typename Enable = void>
struct ASSERT_TYPE_EQ {
    static_assert(sizeof(expected) >= 0, "forces template instantiation");
    static_assert(sizeof(actual__) >= 0, "forces template instantiation");
};
template <typename expected, typename actual__>
struct ASSERT_TYPE_EQ<expected, actual__, typename std::enable_if<!std::is_same<expected, actual__>::value>::type> {
    typename expected::expected_type see_expected_type_above;
    typename actual__::actual_type__ see_actual___type_above;
    static_assert(std::is_same<expected, actual__>::value, "TYPES DON'T MATCH (see types above)");
    static_assert(sizeof(expected) >= 0, "forces template instantiation");
    static_assert(sizeof(actual__) >= 0, "forces template instantiation");
};

#define ASSERT_STATIC_EQ(expected, actual)                                                          \
    ASSERT_TYPE_EQ<std::integral_constant<typename std::decay<decltype(expected)>::type, expected>, \
        std::integral_constant<typename std::decay<decltype(actual)>::type, actual>>{};
