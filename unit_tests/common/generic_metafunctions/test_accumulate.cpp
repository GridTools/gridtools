/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/generic_metafunctions/accumulate.hpp>

#include "gtest/gtest.h"

#include <gridtools/common/array.hpp>
#include <gridtools/common/defs.hpp>

using namespace gridtools;

template <typename... Args>
constexpr bool check_and(Args...) {
    return accumulate(logical_and(), is_array<Args>::type::value...);
}

static_assert(check_and(array<uint_t, 4>{3, 4, 5, 6}, array<int_t, 2>{-2, 3}), "");
static_assert(!check_and(array<uint_t, 4>{3, 4, 5, 6}, array<int_t, 2>{-2, 3}, 7), "");

template <typename... Args>
constexpr bool check_or(Args...) {
    return accumulate(logical_or(), is_array<Args>::type::value...);
}

static_assert(check_or(array<uint_t, 4>{3, 4, 5, 6}, array<int_t, 2>{-2, 3}), "");
static_assert(check_or(array<uint_t, 4>{3, 4, 5, 6}, array<int_t, 2>{-2, 3}), "");
static_assert(check_or(array<uint_t, 4>{3, 4, 5, 6}, array<int_t, 2>{-2, 3}, 7), "");
static_assert(!check_or(-2, 3, 7), "");

TEST(dummy, dummy) {}
