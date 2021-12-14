/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

#include <gridtools/fn/scan.hpp>
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/synthetic.hpp>

namespace gridtools::fn {
    namespace {
        using namespace literals;
        using sid::property;

        struct sum_fold : fwd {
            static GT_FUNCTION consteval auto body() {
                return [](auto acc, auto const &iter) { return acc + *iter; };
            }
        };

        struct sum_scan : fwd {
            static GT_FUNCTION consteval auto body() {
                return scan_pass(
                    [](auto acc, auto const &iter) {
                        return tuple_util::make<tuple>(get<0>(acc) + *iter, get<1>(acc) * *iter);
                    },
                    [](auto acc) { return get<0>(acc); });
            }
        };

        struct sum_fold_with_logues : sum_fold {
            static GT_FUNCTION consteval auto prologue() {
                return tuple_util::make<tuple>([](auto acc, auto const &iter) { return acc + 2 * *iter; });
            }
            static GT_FUNCTION consteval auto epilogue() {
                return tuple_util::make<tuple>([](auto acc, auto const &iter) { return acc + 3 * *iter; });
            }
        };

        struct make_iterator_mock {
            auto operator()() const {
                return [](auto tag, auto const &ptr, auto const &strides) { return at_key<decltype(tag)>(ptr); };
            }
        };

        TEST(scan, smoke) {
            using column_t = int[5];
            using vdim_t = integral_constant<int, 0>;

            column_t a = {0, 0, 0, 0, 0};
            column_t b = {1, 2, 3, 4, 5};
            auto composite = sid::composite::make<integral_constant<int, 0>, integral_constant<int, 1>>(
                sid::synthetic()
                    .set<property::origin>(sid::host_device::make_simple_ptr_holder(&a[0]))
                    .set<property::strides>(tuple_util::make<tuple>(1_c)),
                sid::synthetic()
                    .set<property::origin>(sid::host_device::make_simple_ptr_holder(&b[0]))
                    .set<property::strides>(tuple_util::make<tuple>(1_c)));
            auto ptr = sid::get_origin(composite)();
            auto strides = sid::get_strides(composite);

            {
                column_stage<vdim_t, sum_fold, make_iterator_mock, 0, 1> cs;
                auto res = cs(42, 5, ptr, strides);
                EXPECT_EQ(res, 57);
                for (std::size_t i = 0; i < 5; ++i)
                    EXPECT_EQ(a[i], 0);
            }

            {
                column_stage<vdim_t, sum_scan, make_iterator_mock, 0, 1> cs;
                auto res = cs(tuple_util::make<tuple>(42, 1), 5, ptr, strides);
                EXPECT_EQ(get<0>(res), 57);
                EXPECT_EQ(get<1>(res), 120);
                for (std::size_t i = 0; i < 5; ++i)
                    EXPECT_EQ(a[i], 42 + (i + 1) * (i + 2) / 2);
            }

            {
                column_stage<vdim_t, sum_fold_with_logues, make_iterator_mock, 0, 1> cs;
                auto res = cs(42, 5, ptr, strides);
                EXPECT_EQ(res, 68);
            }
        }

    } // namespace
} // namespace gridtools::fn
