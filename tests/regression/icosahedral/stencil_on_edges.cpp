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

#include <gridtools/stencil_composition/icosahedral.hpp>

#include <icosahedral_regression_fixture.hpp>

#include "neighbours_of.hpp"

using namespace gridtools;
using namespace icosahedral;

struct test_on_edges_functor {
    using in = in_accessor<0, edges, extent<-1, 1, -1, 1>>;
    using out = inout_accessor<1, edges>;
    using param_list = make_param_list<in, out>;
    using location = edges;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        float_type res = 0;
        eval.for_neighbors([&](auto in) { res += in; }, in());
        eval(out()) = res;
    }
};

using stencil_on_edges = regression_fixture<1>;

TEST_F(stencil_on_edges, test) {
    auto in = [](int_t i, int_t j, int_t k, int_t c) { return i + j + k + c; };
    auto ref = [&](int_t i, int_t j, int_t k, int_t c) {
        float_type res = 0;
        for (auto &&item : neighbours_of<edges, edges>(i, j, k, c))
            res += item.call(in);
        return res;
    };
    auto out = make_storage<edges>();
    run_single_stage(test_on_edges_functor(), backend_t(), make_grid(), make_storage<edges>(in), out);
    verify(ref, out);
}
