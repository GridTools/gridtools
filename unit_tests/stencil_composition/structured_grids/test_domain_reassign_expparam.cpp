/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/frontend/expandable_run.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/computation_fixture.hpp>

namespace gridtools {
    struct test_functor {
        using in = in_accessor<0>;
        using out = inout_accessor<1>;
        using param_list = make_param_list<in, out>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    struct fixture : computation_fixture<> {
        fixture() : computation_fixture<>(6, 6, 10) {}
    };

    TEST_F(fixture, run) {
        using storages_t = std::vector<storage_type>;

        std::function<void(storages_t, storages_t)> comp = [grid = make_grid()](storages_t in, storages_t out) {
            expandable_run<2>(
                [](auto in, auto out) {
                    GT_DECLARE_EXPANDABLE_TMP(float_type, tmp);
                    return execute_parallel().stage(test_functor(), in, tmp).stage(test_functor(), tmp, out);
                },
                backend_t(),
                grid,
                in,
                out);
        };

        auto do_test = [&](int n) {
            storages_t in = {3, make_storage([=](int i, int j, int k) { return i + j + k + n; })};
            storages_t out = {3, make_storage()};
            comp(in, out);
            ASSERT_EQ(3, in.size());
            for (size_t i = 0; i != 3; ++i)
                verify(in[i], out[i]);
        };

        do_test(3);
        do_test(7);
    }
} // namespace gridtools
