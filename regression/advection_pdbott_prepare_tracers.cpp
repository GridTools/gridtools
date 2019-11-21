/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <vector>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/expandable_parameters/run.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;

struct prepare_tracers {
    using data = inout_accessor<0>;
    using data_nnow = in_accessor<1>;
    using rho = in_accessor<2>;

    using param_list = make_param_list<data, data_nnow, rho>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(data()) = eval(rho()) * eval(data_nnow());
    }
};

using advection_pdbott_prepare_tracers = regression_fixture<>;

TEST_F(advection_pdbott_prepare_tracers, test) {
    std::vector<storage_type> in, out;

    for (size_t i = 0; i < 11; ++i) {
        out.push_back(make_storage());
        in.push_back(make_storage(i));
    }

    auto comp = [grid = make_grid(), &in, &out, rho = make_const_storage(1.1)] {
        expandable_run<2>(
            [](auto out, auto in, auto rho) { return execute_parallel().stage(prepare_tracers(), out, in, rho); },
            backend_t(),
            grid,
            out,
            in,
            rho);
    };

    comp();
    for (size_t i = 0; i != out.size(); ++i)
        verify([i](int, int, int) { return 1.1 * i; }, out[i]);

    benchmark(comp);
}
