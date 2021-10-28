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

constexpr auto fwd = [](auto count_up, auto count_down, auto prev, auto a, auto b, auto c, auto d) {
    constexpr auto l = [](auto, auto a, auto b, auto c, auto d) {
        return make_tuple(divides(deref(c), deref(b)), divides(deref(d), deref(b)));
    };
    constexpr auto r = [](auto prev, auto a, auto b, auto c, auto d) {
        return make_tuple(divides(deref(c), minus(deref(b), multiplies(deref(a), tuple_get<0>(prev)))),
            divides(minus(deref(d), multiplies(deref(a), tuple_get<1>(prev))),
                minus(deref(b), multiplies(deref(a), tuple_get<0>(prev)))));
    };
    return if_(eq(count_up, 0_c), lambda<l>(prev, a, b, c, d), lambda<r>(prev, a, b, c, d));
    //    return if_(eq(count_up, 0_c), lambda<l>, lambda<r>)(prev, a, b, c, d);
};

constexpr auto bwd = [](auto count_up, auto count_down, auto prev, auto cd) {
    constexpr auto l = [](auto, auto cd) { return tuple_get<1>(deref(cd)); };
    constexpr auto r = [](auto prev, auto cd) {
        return minus(tuple_get<1>(deref(cd)), multiplies(tuple_get<0>(deref(cd)), prev));
    };
    //    return if_(eq(count_up, 0_c), lambda<l>, lambda<r>)(prev, cd);
    return if_(eq(count_up, 0_c), lambda<l>(prev, cd), lambda<r>(prev, cd));
};

constexpr auto tridiag = [](auto a, auto b, auto c, auto d) { return scan<bwd, true>(tlift<scan<fwd>>(a, b, c, d)); };

struct jo {};

constexpr auto wr = [](auto a, auto b, auto c, auto d) { return lambda<fwd>(0_c, 0_c, 0_c, a, b, c, d); };

// using tree_t = ast::parse<wr, double, double, double, double>;

// using res_t = decltype(fwd(0_c,
//    0_c,
//    0_c,
//    ast::in<std::integral_constant<size_t, 0>>(),
//    ast::in<std::integral_constant<size_t, 1>>(),
//    ast::in<std::integral_constant<size_t, 2>>(),
//    ast::in<std::integral_constant<size_t, 3>>()));

// GT_META_PRINT_TYPE(res_t);

constexpr auto foo = [](auto x0, auto x1, auto x2, auto x3) {
    return scan<[](auto x0, auto x1, auto x2, auto x3) {
        return if_(eq(x0, 0_c),
            lambda<[](auto x0, auto x1) { return tuple_get<1>(deref(x1)); }>(x2, x3),
            lambda<[](auto x0, auto x1) {
                return minus(tuple_get<1>(deref(x1)), multiplies(tuple_get<0>(deref(x1)), x0));
            }>(x2, x3));
    },
        true>(tlift<[](auto x0, auto x1, auto x2, auto x3) {
        return scan<[](auto x0, auto x1, auto x2, auto x3, auto x4, auto x5, auto x6) {
            return if_(eq(x0, 0_c),
                lambda<[](auto x0, auto x1, auto x2, auto x3, auto x4) {
                    return make_tuple(divides(deref(x3), deref(x2)), divides(deref(x4), deref(x2)));
                }>(x2, x3, x4, x5, x6),
                lambda<[](auto x0, auto x1, auto x2, auto x3, auto x4) {
                    return make_tuple(divides(deref(x3), minus(deref(x2), multiplies(deref(x1), tuple_get<0>(x0)))),
                        divides(minus(deref(x4), multiplies(deref(x1), tuple_get<1>(x0))),
                            minus(deref(x2), multiplies(deref(x1), tuple_get<0>(x0)))));
                }>(x2, x3, x4, x5, x6));
        }>(x0, x1, x2, x3);
    }>(x0, x1, x2, x3));
};

TEST(tridiag, smoke) {
    using shape_t = double[10][3][3];
    shape_t actual, a, b, c, d;

    using stage_t = make_stage<tridiag, std::identity{}, 0, 1, 2, 3, 4>;
    constexpr auto testee = fencil<naive, stage_t>;
    constexpr auto domain = cartesian(std::tuple(8_c, 8_c, 3_c), std::tuple(1_c, 1_c));
    //    testee(domain, actual, a, b, c, d);

    using tree_t = ast::parse<tridiag, shape_t, shape_t, shape_t, shape_t>;

    std::cout << ast::dump<tridiag, void, void, void, void> << std::endl;
}

// GT_META_PRINT_TYPE(tree_t);