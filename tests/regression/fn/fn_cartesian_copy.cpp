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

#include <fn_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace fn;
    using namespace cartesian;
    using namespace literals;

    struct copy_stencil {
        GT_FUNCTION constexpr auto operator()() const {
            return [](auto const &in) { return deref(in); };
        }
    };

    GT_REGRESSION_TEST(fn_cartesian_copy, test_environment<>, fn_backend_t) {
        auto in = [](int i, int j, int k) { return i + j + k; };
        auto out = TypeParam::make_storage();

        auto apply_copy = [](auto executor, auto &out, auto const &in) {
            executor().arg(out).arg(in).assign(0_c, copy_stencil(), 1_c);
        };

        auto fencil = [&](auto const &sizes, auto &out, auto const &in) {
            auto domain = cartesian_domain(sizes);
            auto backend = make_backend(fn_backend_t(), domain);
            apply_copy(backend.stencil_executor(), out, in);
        };

        auto comp = [&, in = TypeParam::make_const_storage(in)] { fencil(TypeParam::fn_cartesian_sizes(), out, in); };
        comp();
        TypeParam::verify(in, out);
        TypeParam::benchmark("fn_cartesian_copy", comp);
    }
} // namespace
