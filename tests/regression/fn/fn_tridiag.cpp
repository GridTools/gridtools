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

#include "simple_mesh.hpp"

using namespace gridtools;
using namespace literals;
using namespace fn;

constexpr auto fwd0 = [](auto, auto b, auto c, auto d) {
    return make_tuple(divides(deref(c), deref(b)), divides(deref(d), deref(b)));
};

constexpr auto fwd_helper = [](auto prev, auto a, auto c, auto d, auto divisor) {
    return make_tuple(divides(c, divisor), divides(minus(d, multiplies(a, tuple_get<1>(prev))), divisor));
};

constexpr auto fwd = [](auto prev, auto a, auto b, auto c, auto d) {
    return lambda<fwd_helper>(
        prev, deref(a), deref(c), deref(d), minus(deref(b), multiplies(deref(a), tuple_get<0>(prev))));
};

constexpr auto bwd0 = [](auto cd) { return tuple_get<1>(deref(cd)); };

constexpr auto bwd = [](auto prev, auto cd) {
    return minus(tuple_get<1>(deref(cd)), multiplies(tuple_get<0>(deref(cd)), prev));
};

constexpr auto tridiag = [](auto a, auto b, auto c, auto d) {
    return scan_bwd<bwd0, bwd>(tlift<scan_fwd<fwd0, fwd>>(a, b, c, d));
};

using stage_t = make_stage<tridiag, std::identity{}, 0, 1, 2, 3, 4>;
constexpr auto testee = fencil<naive, stage_t>;

constexpr auto I = 3_c;
constexpr auto J = 3_c;
constexpr auto K = 6_c;

using column_t = double[K];

auto solve(column_t const &a, column_t const &b, column_t const &c, column_t const &d) {
    column_t cc, dd;
    cc[0] = c[0] / b[0];
    dd[0] = d[0] / b[0];
    for (size_t k = 1; k != K; ++k) {
        double divisor = b[k] - a[k] * cc[k - 1];
        cc[k] = c[k] / divisor;
        dd[k] = (d[k] - a[k] * dd[k - 1]) / divisor;
    }
    std::array<double, K> res;
    res[K - 1] = dd[K - 1];
    for (int k = K - 2; k >= 0; --k)
        res[k] = dd[k] - cc[k] * res[k + 1];
    return res;
};

TEST(tridiag, catresian) {
    using shape_t = column_t[I][J];
    shape_t actual, a, b, c, d;

    for (size_t i = 0; i != I; ++i)
        for (size_t j = 0; j != J; ++j)
            for (size_t k = 0; k != K; ++k) {
                a[i][j][k] = std::rand();
                b[i][j][k] = std::rand();
                c[i][j][k] = std::rand();
                d[i][j][k] = std::rand();
            }

    testee(cartesian(std::tuple(I, J, K)), actual, a, b, c, d);

    for (size_t i = 0; i != I; ++i)
        for (size_t j = 0; j != J; ++j) {
            auto expected = solve(a[i][j], b[i][j], c[i][j], d[i][j]);
            for (size_t k = 0; k != K; ++k)
                EXPECT_DOUBLE_EQ(actual[i][j][k], expected[k]) << "i=" << i << ", j=" << j << ", k=" << k;
        }
}

TEST(tridiag, unstructured) {
    using namespace simple_mesh;
    using shape_t = column_t[n_vertices];
    shape_t actual, a, b, c, d;
    for (size_t h = 0; h != n_vertices; ++h)
        for (size_t k = 0; k != K; ++k) {
            a[h][k] = std::rand();
            b[h][k] = std::rand();
            c[h][k] = std::rand();
            d[h][k] = std::rand();
        }

    testee(unstructured(std::tuple(n_vertices, K)), actual, a, b, c, d);

    for (size_t h = 0; h != n_vertices; ++h) {
        auto expected = solve(a[h], b[h], c[h], d[h]);
        for (size_t k = 0; k != K; ++k)
            EXPECT_DOUBLE_EQ(actual[h][k], expected[k]) << "h=" << h << ", k=" << k;
    }
}
