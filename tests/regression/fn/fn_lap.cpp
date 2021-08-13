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

#include <gtest/gtest.h>

#include <gridtools/fn.hpp>

using namespace gridtools;
using namespace fn;
using namespace literals;

inline constexpr auto ldif = [](auto d) { return [s = shift(d, -1_c)](auto in) { return deref(in) - deref(s(in)); }; };

inline constexpr auto rdif = [](auto d) { return [=](auto in) { return ldif(d)(shift(d, 1_c)(in)); }; };

inline constexpr auto dif2 = [](auto d) { return [=](auto in) { return ldif(d)(lift(rdif(d))(in)); }; };

inline constexpr auto lap = [](auto const &in) { return dif2(i)(in) + dif2(j)(in); };

TEST(fn, lap) {
    double actual[10][10][3];
    double in[10][10][3];
    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
            for (int k = 0; k < 3; ++k)
                in[i][j][k] = std::rand();

    auto expected = [&](auto i, auto j, auto k) {
        return in[i + 1][j][k] + in[i - 1][j][k] + in[i][j + 1][k] + in[i][j - 1][k] - 4 * in[i][j][k];
    };

    fencil(naive(), closure(cartesian(std::tuple(8_c, 8_c, 3_c), std::array{1_c, 1_c}), lap, out(actual), in));

    for (int i = 1; i < 9; ++i)
        for (int j = 1; j < 9; ++j)
            for (int k = 0; k < 3; ++k)
                EXPECT_DOUBLE_EQ(actual[i][j][k], expected(i, j, k)) << "i=" << i << ", j=" << j << ", k=" << k;
}
