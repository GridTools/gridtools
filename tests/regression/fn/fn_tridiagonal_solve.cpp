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

#include <gridtools/fn/cartesian2.hpp>
#include <gridtools/fn/unstructured2.hpp>

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace fn;
    using namespace literals;

    struct forward_scan : fwd {
        static GT_FUNCTION constexpr auto prologue() {
            return tuple(scan_pass(
                [](auto /*acc*/, auto const & /*a*/, auto const &b, auto const &c, auto const &d) {
                    return tuple(deref(c) / deref(b), deref(d) / deref(b));
                },
                identity()));
        }

        static GT_FUNCTION constexpr auto body() {
            return scan_pass(
                [](auto acc, auto const &a, auto const &b, auto const &c, auto const &d) {
                    auto [cp, dp] = acc;
                    return tuple(
                        deref(c) / (deref(b) - deref(a) * cp), (deref(d) - deref(a) * dp) / (deref(b) - deref(a) * cp));
                },
                identity());
        }
    };

    struct backward_scan : bwd {
        static GT_FUNCTION constexpr auto prologue() {
            return tuple(scan_pass(
                [](auto /*xp*/, auto const &cpdp) {
                    auto [cp, dp] = deref(cpdp);
                    return dp;
                },
                identity()));
        }

        static GT_FUNCTION constexpr auto body() {
            return scan_pass(
                [](auto xp, auto const &cpdp) {
                    auto [cp, dp] = deref(cpdp);
                    return dp - cp * xp;
                },
                identity());
        }
    };

    GT_ENVIRONMENT_TEST_SUITE(fn_cartesian_tridiagonal_solve,
        vertical_test_environment<>,
        fn_backend_t,
        (double, inlined_params<12, 33, 6>),
        (double, inlined_params<23, 11, 6>));
    GT_ENVIRONMENT_TYPED_TEST(fn_cartesian_tridiagonal_solve, test) {
        using float_t = typename TypeParam::float_t;

        auto a = [](int, int, int) -> float_t { return -1; };
        auto b = [](int, int, int) -> float_t { return 3; };
        auto c = [](int, int, int) -> float_t { return 1; };
        auto d = [](int, int, int k) -> float_t { return k == 0 ? 4 : k == 5 ? 2 : 3; };
        auto cpdp = [&](int i, int j, int k) {
            auto rec = [&](auto &&rec, int i, int j, int k) -> tuple<float_t, float_t> {
                if (k == 0)
                    return {c(i, j, k) / b(i, j, k), d(i, j, k) / b(i, j, k)};
                auto [cp, dp] = rec(rec, i, j, k - 1);
                return {c(i, j, k) / (b(i, j, k) - a(i, j, k) * cp),
                    (d(i, j, k) - a(i, j, k) * dp) / (b(i, j, k) - a(i, j, k) * cp)};
            };
            return rec(rec, i, j, k);
        };
        auto expected = [&, kmax = TypeParam::d(2) - 1](int i, int j, int k) {
            auto rec = [&](auto &&rec, int i, int j, int k) -> float_t {
                auto [cp, dp] = cpdp(i, j, k);
                if (k == kmax)
                    return dp;
                return dp - cp * rec(rec, i, j, k + 1);
            };
            return rec(rec, i, j, k);
        };

        auto tridiagonal_solve =
            [](auto executor, auto const &a, auto const &b, auto const &c, auto const &d, auto &cpdp, auto &x) {
                return executor()
                    .arg(a)
                    .arg(b)
                    .arg(c)
                    .arg(d)
                    .arg(cpdp)
                    .arg(x)
                    .assign(4_c, forward_scan(), tuple<float_t, float_t>(0, 0), 0_c, 1_c, 2_c, 3_c)
                    .assign(5_c, backward_scan(), float_t(0), 4_c);
            };

        auto tridiagonal_solve_fencil =
            [&](auto sizes, auto const &a, auto const &b, auto const &c, auto const &d, auto &x) {
                auto domain = cartesian_domain(sizes);
                auto backend = make_backend(fn_backend_t(), domain);
                auto cpdp = backend.template make_tmp<tuple<float_t, float_t>>();
                tridiagonal_solve(backend.vertical_executor(), a, b, c, d, cpdp, x);
            };

        auto x = TypeParam::make_storage();
        auto comp = [&,
                        a = TypeParam::make_const_storage(a),
                        b = TypeParam::make_const_storage(b),
                        c = TypeParam::make_const_storage(c),
                        d = TypeParam::make_const_storage(d)] {
            tridiagonal_solve_fencil(TypeParam::fn_cartesian_sizes(), a, b, c, d, x);
        };
        comp();
        TypeParam::verify(expected, x);
    }
} // namespace
