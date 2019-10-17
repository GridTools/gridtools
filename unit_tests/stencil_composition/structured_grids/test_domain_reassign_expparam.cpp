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

#include <gridtools/stencil_composition/expandable_parameters/make_computation.hpp>
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
            arg<0, storages_t> p_in;
            arg<1, storages_t> p_out;
            tmp_arg<2, std::vector<float_type>> p_tmp;
            expandable_compute<backend_t>(expand_factor<2>(),
                grid,
                p_in = in,
                p_out = out,
                make_multistage(
                    execute::forward(), make_stage<test_functor>(p_in, p_tmp), make_stage<test_functor>(p_tmp, p_out)));
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
