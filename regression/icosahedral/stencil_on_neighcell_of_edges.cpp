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

template <uint_t>
struct test_on_cells_functor {
    GT_DEFINE_ACCESSORS(
        GT_IN_ACCESSOR(in, enumtype::cells, extent<1, -1, 1, -1>), GT_INOUT_ACCESSOR(out, enumtype::edges));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out{}) = eval(on_cells(binop::sum{}, float_type{}, in{}));
    }
};

using stencil_on_neighcell_of_edges = regression_fixture<1>;

TEST_F(stencil_on_neighcell_of_edges, test) {
    auto in = [](int_t i, int_t c, int_t j, int_t k) { return i + c + j + k; };
    auto ref = [&](int_t i, int_t c, int_t j, int_t k) {
        float_type res = {};
        for (auto &&item : neighbours_of<edges, cells>(i, c, j, k))
            res += item.call(in);
        return res;
    };
    arg<0, cells> p_in;
    arg<1, edges> p_out;
    auto out = make_storage<edges>();
    auto comp = make_computation(p_in = make_storage<cells>(in),
        p_out = out,
        make_multistage(execute::forward(), make_stage<test_on_cells_functor, topology_t, edges>(p_in, p_out)));
    comp.run();
    verify(make_storage<edges>(ref), out);
    benchmark(comp);
}
