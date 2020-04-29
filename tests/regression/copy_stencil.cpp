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

#include <gridtools/stencil/cartesian.hpp>

#include <stencil_select.hpp>
#include <test_environment.hpp>

namespace {
    using namespace gridtools;
    using namespace stencil;
    using namespace cartesian;

    struct copy_functor {
        using in = in_accessor<0>;
        using out = inout_accessor<1>;

        using param_list = make_param_list<in, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            eval(out()) = eval(in());
        }
    };

    GT_REGRESSION_TEST(copy_stencil, test_environment<>, stencil_backend_t) {
        auto in = [](int i, int j, int k) { return i + j + k; };
        std::vector<decltype(TypeParam::make_const_storage(in))> in_storages;
        std::vector<decltype(TypeParam::make_storage())> out_storages;
        for (std::size_t set = 0; set < 16; ++set) {
            in_storages.push_back(TypeParam::make_const_storage(in));
            out_storages.push_back(TypeParam::make_storage());
        }
        std::size_t run = 0;
        auto comp = [&, grid = TypeParam::make_grid()] {
            run_single_stage(copy_functor(),
                stencil_backend_t(),
                grid,
                in_storages[run % in_storages.size()],
                out_storages[run % out_storages.size()]);
            ++run;
        };
        comp();
        TypeParam::verify(in, out_storages[0]);
        TypeParam::benchmark("copy_stencil", comp);
    }
} // namespace