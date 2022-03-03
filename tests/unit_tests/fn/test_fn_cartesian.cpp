/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/cartesian2.hpp>

#include <gtest/gtest.h>

#include <gridtools/fn/backend2/naive.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;

        struct stencil {
            constexpr auto operator()() const {
                using namespace cartesian::dim;
                return [](auto const &in) { return deref(shift(in, i(), 1_c)); };
            }
        };

        struct fwd_sum_scan : fwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass(
                    [](auto acc, auto const &iter) { return acc + deref(iter); }, [](auto acc) { return acc; });
            }
        };

        struct bwd_sum_scan : bwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass(
                    [](auto acc, auto const &iter) { return acc + deref(iter); }, [](auto acc) { return acc; });
            }
        };

        TEST(cartesian, stencil) {
            auto apply_stencil = [](auto executor, auto &out, auto const &in) {
                executor().arg(out).arg(in).assign(0_c, stencil(), 1_c);
            };

            auto fencil = [&](auto const &sizes, auto &out, auto const &in) {
                auto domain = cartesian_domain(sizes);
                auto backend = make_backend(backend::naive(), domain);
                auto tmp = backend.template make_tmp<int>();
                auto compute_domain = cartesian_domain(std::array{sizes[0] - 1, sizes[1], sizes[2]});
                auto compute_backend = make_backend(backend::naive(), compute_domain);
                apply_stencil(compute_backend.stencil_executor(), tmp, in);
                apply_stencil(compute_backend.stencil_executor(), out, tmp);
            };

            int in[5][3][2], out[5][3][2] = {};
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 2; ++k)
                        in[i][j][k] = 6 * i + 2 * j + k;

            fencil(std::array{5, 3, 2}, out, in);

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 2; ++k)
                        EXPECT_EQ(out[i][j][k], 6 * (i + 2) + 2 * j + k);
        }

        TEST(cartesian, vertical) {
            auto apply_double_scan = [](auto executor, auto &a, auto &b, auto const &c) {
                executor()
                    .arg(a)
                    .arg(b)
                    .arg(c)
                    .assign(1_c, fwd_sum_scan(), 42, 2_c)
                    .assign(0_c, bwd_sum_scan(), 8, 1_c);
            };

            auto double_scan = [&](auto sizes, auto &a, auto &b, auto const &c) {
                auto domain = cartesian_domain(sizes);
                auto backend = make_backend(backend::naive(), domain);
                apply_double_scan(backend.vertical_executor(), a, b, c);
            };

            std::array<int, 3> sizes = {5, 3, 2};
            int a[5][3][2] = {}, b[5][3][2] = {}, c[5][3][2];
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 2; ++k)
                        c[i][j][k] = 6 * i + 2 * j + k;

            double_scan(sizes, a, b, c);

            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 3; ++j) {
                    int res = 42;
                    for (int k = 0; k < 2; ++k) {
                        res += c[i][j][k];
                        EXPECT_EQ(b[i][j][k], res);
                    }
                    res = 8;
                    for (int k = 1; k >= 0; --k) {
                        res += b[i][j][k];
                        EXPECT_EQ(a[i][j][k], res);
                    }
                }
        }
    } // namespace
} // namespace gridtools::fn
