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

#include <gridtools/common/binops.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "neighbours_of.hpp"

using namespace gridtools;

struct test_on_edges_functor {
    using in = in_accessor<0, enumtype::edges, extent<0, 1, 0, 1>>;
    using out = inout_accessor<1, enumtype::cells>;
    using param_list = make_param_list<in, out>;
    using location = enumtype::cells;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out{}) = eval(on_edges(binop::sum{}, float_type{}, in{}));
    }
};

struct test_on_cells_functor {
    using in = in_accessor<0, enumtype::cells, extent<-1, 1, -1, 1>>;
    using out = inout_accessor<1, enumtype::cells>;
    using param_list = make_param_list<in, out>;
    using location = enumtype::cells;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out{}) = eval(on_cells(binop::sum{}, float_type{}, in{}));
    }
};

using stencil_fused = regression_fixture<2>;

TEST_F(stencil_fused, test) {
    arg<0, edges> p_in;
    arg<1, cells> p_out;
    tmp_arg<0, cells> p_tmp;

    auto in = [](int_t i, int_t c, int_t j, int_t k) { return i + c + j + k; };

    auto tmp = [&](int_t i, int_t c, int_t j, int_t k) {
        float_type res{};
        for (auto &&item : neighbours_of<cells, edges>(i, c, j, k))
            res += item.call(in);
        return res;
    };

    auto ref = [&](int_t i, int_t c, int_t j, int_t k) {
        float_type res{};
        for (auto &&item : neighbours_of<cells, cells>(i, c, j, k))
            res += item.call(tmp);
        return res;
    };

    auto out = make_storage<cells>();

    make_computation(p_in = make_storage<edges>(in),
        p_out = out,
        make_multistage(execute::forward(),
            make_stage<test_on_edges_functor>(p_in, p_tmp),
            make_stage<test_on_cells_functor>(p_tmp, p_out)))
        .run();

    verify(make_storage<cells>(ref), out);
}
