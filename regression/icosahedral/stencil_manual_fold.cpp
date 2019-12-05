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

#include <gridtools/common/array.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "neighbours_of.hpp"

using namespace gridtools;
using namespace expressions;

using weight_edges_t = array<float_type, 3>;

struct test_on_edges_functor {
    using cell_area = in_accessor<0, enumtype::cells, extent<-1, 1, -1, 1>>;
    using weight_edges = inout_accessor<1, enumtype::cells>;
    using param_list = make_param_list<cell_area, weight_edges>;
    using location = enumtype::cells;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        auto &&out = eval(weight_edges());
        auto focus = eval(cell_area());
        int i = 0;
        eval.for_neighbors([&](auto neighbor) { out[i++] = neighbor / focus; }, cell_area());
    }
};

using stencil_manual_fold = regression_fixture<1>;

TEST_F(stencil_manual_fold, test) {
    auto in = [](int_t i, int_t j, int_t k, int_t c) { return 1. + i + j + k + c; };
    auto ref = [&](int_t i, int_t j, int_t k, int_t c) -> weight_edges_t {
        auto val = [&](int e) -> float_type {
            return neighbours_of<cells, cells>(i, j, k, c)[e].call(in) / in(i, j, k, c);
        };
        return {val(0), val(1), val(2)};
    };
    auto out = make_storage<cells, weight_edges_t>();
    auto comp = [&] { easy_run(test_on_edges_functor(), backend_t(), make_grid(), make_storage<cells>(in), out); };
    comp();
    verify(ref, out);
    benchmark(comp);
}
