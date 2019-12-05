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

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        auto &&res = eval(out());
        res = 0;
        eval.for_neighbors([&](auto in) { res += in; }, in());
    }
};

struct test_on_cells_functor {
    using in = in_accessor<0, enumtype::cells, extent<-1, 1, -1, 1>>;
    using out = inout_accessor<1, enumtype::cells>;
    using param_list = make_param_list<in, out>;
    using location = enumtype::cells;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        auto &&res = eval(out());
        res = 0;
        eval.for_neighbors([&](auto in) { res += in; }, in());
    }
};

const auto spec = [](auto in, auto out) {
    GT_DECLARE_COLORED_TMP(float_type, enumtype::cells, tmp);
    return execute_parallel().stage(test_on_edges_functor(), in, tmp).stage(test_on_cells_functor(), tmp, out);
};

using stencil_fused = regression_fixture<2>;

TEST_F(stencil_fused, test) {
    auto in = [](int_t i, int_t j, int_t k, int_t c) { return i + j + k + c; };

    auto tmp = [&](int_t i, int_t j, int_t k, int_t c) {
        float_type res{};
        for (auto &&item : neighbours_of<cells, edges>(i, j, k, c))
            res += item.call(in);
        return res;
    };

    auto ref = [&](int_t i, int_t j, int_t k, int_t c) {
        float_type res{};
        for (auto &&item : neighbours_of<cells, cells>(i, j, k, c))
            res += item.call(tmp);
        return res;
    };

    auto out = make_storage<cells>();
    run(spec, backend_t(), make_grid(), make_storage<edges>(in), out);
    verify(ref, out);
}
