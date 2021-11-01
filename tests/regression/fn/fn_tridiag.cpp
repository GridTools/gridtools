/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstdlib>
#include <utility>

#include <gtest/gtest.h>

#include <gridtools/fn.hpp>
#include <gridtools/fn/backend/naive.hpp>

using namespace gridtools;
using namespace literals;
using namespace fn;

constexpr auto fwd0 = [](auto, auto b, auto c, auto d) { return make_tuple(divides(c, b), divides(d, b)); };

constexpr auto fwd = [](auto prev, auto a, auto b, auto c, auto d) {
    constexpr auto tmp = [](auto prev, auto a, auto c, auto d, auto divisor) {
        return make_tuple(divides(c, divisor), divides(minus(d, multiplies(a, tuple_get<1>(prev))), divisor));
    };
    return lambda<tmp>(prev, deref(a), deref(c), deref(d), minus(deref(b), multiplies(deref(a), tuple_get<0>(prev))));
};

constexpr auto bwd0 = [](auto cd) { return tuple_get<1>(deref(cd)); };

constexpr auto bwd = [](auto prev, auto cd) {
    return minus(tuple_get<1>(deref(cd)), multiplies(tuple_get<0>(deref(cd)), prev));
};

constexpr auto tridiag = [](auto a, auto b, auto c, auto d) {
    return scan_bwd.pass<bwd>.prologue<bwd0>(tlift<scan_fwd.pass<fwd>.prologue<fwd0>>(a, b, c, d));
};

TEST(tridiag, smoke) {
    using shape_t = double[10][3][3];
    shape_t actual, a, b, c, d;

    using stage_t = make_stage<tridiag, std::identity{}, 0, 1, 2, 3, 4>;
    constexpr auto testee = fencil<naive, stage_t>;
    constexpr auto domain = cartesian(std::tuple(8_c, 8_c, 3_c), std::tuple(1_c, 1_c));
    //    testee(domain, actual, a, b, c, d);

    std::cout << ast::dump<"tridiag", tridiag, void, void, void, void> << std::endl;
}

// GT_META_PRINT_TYPE(tree_t);