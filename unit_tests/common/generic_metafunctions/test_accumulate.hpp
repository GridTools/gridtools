/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "gtest/gtest.h"
#include <gridtools/common/array.hpp>
#include <gridtools/common/defs.hpp>
#include <gridtools/common/generic_metafunctions/accumulate.hpp>
#include <gridtools/common/generic_metafunctions/binary_ops.hpp>

using namespace gridtools;

template <typename... Args>
GT_FUNCTION static constexpr bool check_or(Args... args) {
    return accumulate(logical_or(), is_array<Args>::type::value...);
}

template <typename... Args>
GT_FUNCTION static constexpr bool check_and(Args... args) {
    return accumulate(logical_and(), is_array<Args>::type::value...);
}

GT_FUNCTION
static bool test_accumulate_and() {
    GT_STATIC_ASSERT((check_and(array<uint_t, 4>{3, 4, 5, 6}, array<int_t, 2>{-2, 3})), "Error");
    GT_STATIC_ASSERT((!check_and(array<uint_t, 4>{3, 4, 5, 6}, array<int_t, 2>{-2, 3}, 7)), "Error");

    return true;
}

GT_FUNCTION
static bool test_accumulate_or() {

    GT_STATIC_ASSERT((check_or(array<uint_t, 4>{3, 4, 5, 6}, array<int_t, 2>{-2, 3})), "Error");
    GT_STATIC_ASSERT((check_or(array<uint_t, 4>{3, 4, 5, 6}, array<int_t, 2>{-2, 3})), "Error");
    GT_STATIC_ASSERT((check_or(array<uint_t, 4>{3, 4, 5, 6}, array<int_t, 2>{-2, 3}, 7)), "Error");
    GT_STATIC_ASSERT((!check_or(-2, 3, 7)), "Error");

    return true;
}
