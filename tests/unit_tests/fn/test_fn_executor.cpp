/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/executor.hpp>

#include <gtest/gtest.h>

#include <gridtools/fn/backend2/naive.hpp>
#include <gridtools/fn/scan.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;
        using sid::property;

        template <int I>
        using int_t = integral_constant<int, I>;

        struct stencil {
            GT_FUNCTION constexpr auto operator()() const {
                return [](auto const &iter) { return 2 * *iter; };
            }
        };

        struct fwd_sum_scan : fwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass([](auto acc, auto const &iter) { return acc + *iter; }, [](auto acc) { return acc; });
            }
        };

        struct bwd_sum_scan : bwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass([](auto acc, auto const &iter) { return acc + *iter; }, [](auto acc) { return acc; });
            }
        };

        struct make_iterator_mock {
            GT_FUNCTION auto operator()() const {
                return [](auto tag, auto const &ptr, auto const &strides) { return at_key<decltype(tag)>(ptr); };
            }
        };

        TEST(stencil_executor, smoke) {
            using backend_t = backend::naive;
            auto domain = hymap::keys<int_t<0>, int_t<1>>::values(2_c, 3_c);

            auto alloc = backend::tmp_allocator(backend_t());
            auto a = backend::allocate_global_tmp<int>(alloc, domain);
            auto b = backend::allocate_global_tmp<int>(alloc, domain);
            auto c = backend::allocate_global_tmp<int>(alloc, domain);

            auto loop = [](auto x, auto f) {
                auto ptr = sid::get_origin(x)();
                auto strides = sid::get_strides(x);
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        f(ptr, i, j);
                        sid::shift(ptr, sid::get_stride<int_t<1>>(strides), 1_c);
                    }
                    sid::shift(ptr, sid::get_stride<int_t<1>>(strides), -3_c);
                    sid::shift(ptr, sid::get_stride<int_t<0>>(strides), 1_c);
                }
            };
            loop(a, [](auto &ptr, int i, int j) { *ptr = 0; });
            loop(b, [](auto &ptr, int i, int j) { *ptr = 0; });
            loop(c, [](auto &ptr, int i, int j) { *ptr = 3 * i + j; });

            {
                stencil_executor(backend_t(), make_iterator_mock(), domain)
                    .arg(a)
                    .arg(b)
                    .arg(c)
                    .assign(1_c, stencil(), 2_c)
                    .assign(0_c, stencil(), 1_c);
            }

            loop(a, [](auto const &ptr, int i, int j) { EXPECT_EQ(*ptr, (3 * i + j) * 4); });
            loop(b, [](auto const &ptr, int i, int j) { EXPECT_EQ(*ptr, (3 * i + j) * 2); });
            loop(c, [](auto const &ptr, int i, int j) { EXPECT_EQ(*ptr, (3 * i + j) * 1); });
        }

        TEST(vertical_executor, smoke) {
            using backend_t = backend::naive;
            using stages_specs_t = meta::list<column_stage<int_t<1>, fwd_sum_scan, make_iterator_mock, 1, 2>,
                column_stage<int_t<1>, bwd_sum_scan, make_iterator_mock, 0, 1>>;
            auto domain = hymap::keys<int_t<0>, int_t<1>>::values(2_c, 3_c);

            auto alloc = backend::tmp_allocator(backend_t());
            auto a = backend::allocate_global_tmp<int>(alloc, domain);
            auto b = backend::allocate_global_tmp<int>(alloc, domain);
            auto c = backend::allocate_global_tmp<int>(alloc, domain);

            auto loop = [](auto x, auto f) {
                auto ptr = sid::get_origin(x)();
                auto strides = sid::get_strides(x);
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        f(ptr, i, j);
                        sid::shift(ptr, sid::get_stride<int_t<1>>(strides), 1_c);
                    }
                    sid::shift(ptr, sid::get_stride<int_t<1>>(strides), -3_c);
                    sid::shift(ptr, sid::get_stride<int_t<0>>(strides), 1_c);
                }
            };
            loop(a, [](auto &ptr, int i, int j) { *ptr = 0; });
            loop(b, [](auto &ptr, int i, int j) { *ptr = 0; });
            loop(c, [](auto &ptr, int i, int j) { *ptr = 3 * i + j; });

            {
                vertical_executor(backend_t(), make_iterator_mock(), domain, int_t<1>())
                    .arg(a)
                    .arg(b)
                    .arg(c)
                    .assign(1_c, fwd_sum_scan(), 42, 2_c)
                    .assign(0_c, bwd_sum_scan(), 8, 1_c);
            }

            loop(a, [](auto const &ptr, int i, int j) {
                EXPECT_EQ(*ptr, -(9 * i * j * (j + 1) - 108 * i + j * j * j + 251 * j - 828) / 6);
            });
            loop(b, [](auto const &ptr, int i, int j) { EXPECT_EQ(*ptr, 42 + 3 * i * (j + 1) + j * (j + 1) / 2); });
            loop(c, [](auto const &ptr, int i, int j) { EXPECT_EQ(*ptr, 3 * i + j); });
        }
    } // namespace
} // namespace gridtools::fn
