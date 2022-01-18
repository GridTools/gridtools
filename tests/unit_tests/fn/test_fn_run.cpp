/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/fn/run.hpp>

#include <tuple>
#include <variant>

#include <gtest/gtest.h>

#include <gridtools/fn/backend2/naive.hpp>
#include <gridtools/fn/scan.hpp>
#include <gridtools/fn/stencil_stage.hpp>
#include <gridtools/sid/concept.hpp>

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

        struct sum_scan : fwd {
            static GT_FUNCTION constexpr auto body() {
                return scan_pass([](auto acc, auto const &iter) { return acc + *iter; }, [](auto acc) { return acc; });
            }
        };

        struct make_iterator_mock {
            GT_FUNCTION auto operator()() const {
                return [](auto tag, auto const &ptr, auto const &strides) { return at_key<decltype(tag)>(ptr); };
            }
        };

        TEST(run, stencils) {
            using backend_t = backend::naive;
            using stages_specs_t = meta::list<stencil_stage<stencil, make_iterator_mock, 1, 2>,
                stencil_stage<stencil, make_iterator_mock, 0, 1>>;
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

            run(backend_t(),
                int_t<1>(),
                stages_specs_t(),
                domain,
                std::forward_as_tuple(a, b, c),
                std::forward_as_tuple(std::monostate(), std::monostate()));

            loop(a, [](auto const &ptr, int i, int j) { EXPECT_EQ(*ptr, (3 * i + j) * 4); });
            loop(b, [](auto const &ptr, int i, int j) { EXPECT_EQ(*ptr, (3 * i + j) * 2); });
            loop(c, [](auto const &ptr, int i, int j) { EXPECT_EQ(*ptr, (3 * i + j) * 1); });
        }

        TEST(run, scan_and_stencil) {
            using backend_t = backend::naive;
            using stages_specs_t = meta::list<column_stage<int_t<1>, sum_scan, make_iterator_mock, 1, 2>,
                stencil_stage<stencil, make_iterator_mock, 0, 1>>;
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

            run(backend_t(),
                int_t<1>(),
                stages_specs_t(),
                domain,
                std::forward_as_tuple(a, b, c),
                std::forward_as_tuple(42, std::monostate()));

            loop(a,
                [](auto const &ptr, int i, int j) { EXPECT_EQ(*ptr, (42 + (3 * i * (j + 1) + j * (j + 1) / 2)) * 2); });
            loop(b, [](auto const &ptr, int i, int j) { EXPECT_EQ(*ptr, 42 + (3 * i * (j + 1) + j * (j + 1) / 2)); });
            loop(c, [](auto const &ptr, int i, int j) { EXPECT_EQ(*ptr, 3 * i + j); });
        }

    } // namespace
} // namespace gridtools::fn
