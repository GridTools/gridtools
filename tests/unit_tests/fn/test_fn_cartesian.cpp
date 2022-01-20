/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <iostream>

#include <gridtools/fn/backend2/naive.hpp>
#include <gridtools/fn/cartesian2.hpp>
#include <gridtools/stencil/positional.hpp>

#include <gtest/gtest.h>

namespace gridtools::fn {
    namespace {
        using namespace literals;

        struct copy_stencil {
            constexpr auto operator()() const {
                return [](auto const &in) { return deref(in); };
            }
        };

        TEST(cartesian, smoke) {
            auto apply_copy_stencil = [](auto executor, auto &out, auto const &in) {
                executor().arg(out).arg(in).assign(0_c, copy_stencil(), 1_c);
            };

            auto copy_fencil = [&](auto sizes, auto &out, auto const &in) {
                auto domain = cartesian(sizes);
                auto backend = make_backend(backend::naive(), domain);
                auto tmp = backend.make_tmp_like(out);
                apply_copy_stencil(backend.stencil_executor(), tmp, in);
                apply_copy_stencil(backend.stencil_executor(), out, tmp);
            };

            std::array<int, 3> sizes = {5, 3, 2};
            int in[5][3][2], out[5][3][2] = {};
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 2; ++k)
                        in[i][j][k] = 6 * i + 2 * j + k;

            copy_fencil(sizes, out, in);

            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 2; ++k)
                        EXPECT_EQ(out[i][j][k], 6 * i + 2 * j + k);
        }
    } // namespace
} // namespace gridtools::fn
