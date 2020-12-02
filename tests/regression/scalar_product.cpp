/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/reduction.hpp>
#include <gridtools/stencil/cartesian.hpp>

#include <reduction_select.hpp>
#include <test_environment.hpp>
#include <verifier.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct mul_functor {
        using out = inout_accessor<0>;
        using lhs = in_accessor<1>;
        using rhs = in_accessor<2>;

        using param_list = make_param_list<out, lhs, rhs>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out()) = eval(lhs()) * eval(rhs());
        }
    };

    int m(int x) { return x % 2 ? 1 : -1; }

    GT_REGRESSION_TEST(scalar_product, test_environment<>, reduction_backend_t) {
        auto lhs = [](int i, int, int k) { return m(i); };
        auto rhs = [](int, int j, int) { return m(j); };
        auto out = reduction::make_reducible<reduction_backend_t, storage_traits_t>(
            float_t(0), TypeParam::d(0), TypeParam::d(1), TypeParam::d(2));
        auto comp = [&out,
                        grid = TypeParam::make_grid(),
                        lhs = TypeParam::make_const_storage(lhs),
                        rhs = TypeParam::make_const_storage(rhs)] {
            run_single_stage(mul_functor(), stencil_backend_t(), grid, out, lhs, rhs);
            return out.reduce(reduction::plus());
        };
        comp();
        double expected = 0;
        for (size_t i = 0; i < TypeParam::d(0); ++i)
            for (size_t j = 0; j < TypeParam::d(1); ++j)
                for (size_t k = 0; k < TypeParam::d(2); ++k)
                    expected += lhs(i, j, k) * rhs(i, j, k);
        EXPECT_NEAR(comp(), expected, default_precision<float_t>());
        TypeParam::benchmark("scalar_product", comp);
    }
} // namespace
