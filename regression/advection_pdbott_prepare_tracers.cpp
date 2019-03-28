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

#include <gridtools/stencil_composition/expandable_parameters/make_computation.hpp>
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
    using storages_t = std::vector<storage_type>;

    arg<0, storages_t> p_out;
    arg<1, storages_t> p_in;
    arg<2, storage_type> p_rho;

    storages_t in, out;

    for (size_t i = 0; i < 11; ++i) {
        out.push_back(make_storage());
        in.push_back(make_storage(1. * i));
    }

    auto comp = gridtools::make_expandable_computation<backend_t>(expand_factor<2>(),
        make_grid(),
        p_out = out,
        p_in = in,
        p_rho = make_storage(1.1),
        make_multistage(execute::forward(), make_stage<prepare_tracers>(p_out, p_in, p_rho)));

    comp.run();
    for (size_t i = 0; i != out.size(); ++i)
        verify(make_storage([i](int_t, int_t, int_t) { return 1.1 * i; }), out[i]);

    benchmark(comp);
}
