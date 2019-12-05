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
    using in = in_accessor<0, enumtype::edges, extent<1, -1, 1, -1>>;
    using out = inout_accessor<1, enumtype::cells>;
    using param_list = make_param_list<in, out>;
    using location = enumtype::cells;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        float_type res = 0;
        eval.for_neighbors([&res](auto in) { res += in; }, in());
        eval(out()) = res;
    }
};

using stencil_on_neighedge_of_cells = regression_fixture<1>;

TEST_F(stencil_on_neighedge_of_cells, Test) {
    auto in = [](int_t i, int_t c, int_t j, int_t k) { return i + c + j + k; };
    auto ref = [&](int_t i, int_t c, int_t j, int_t k) {
        float_type res = {};
        for (auto &&item : neighbours_of<cells, edges>(i, c, j, k))
            res += item.call(in);
        return res;
    };
    auto out = make_storage<cells>();
    easy_run(test_on_edges_functor(), backend_t(), make_grid(), make_storage<edges>(in), out);
    verify(ref, out);
}
