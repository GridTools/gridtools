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
#include <gridtools/tools/icosahedral_regression_fixture.hpp>

#include "div_functors.hpp"
#include "operators_repository.hpp"

using namespace gridtools;
using namespace ico_operators;

struct div : regression_fixture<2> {
    operators_repository repo = {d(0), d(1)};
};

TEST_F(div, reduction_into_scalar) {
    auto spec = [](auto in_edges, auto edge_length, auto cell_area_reciprocal, auto out) {
        GT_DECLARE_ICO_TMP((array<float_type, 3>), cells, weights);
        return execute_parallel()
            .ij_cached(weights)
            .stage(div_prep_functor(), edge_length, cell_area_reciprocal, weights)
            .stage(div_functor_reduction_into_scalar(), in_edges, weights, out);
    };
    auto out = make_storage<cells>();
    run(spec,
        backend_t(),
        make_grid(),
        make_storage<edges>(repo.u),
        make_storage<edges>(repo.edge_length),
        make_storage<cells>(repo.cell_area_reciprocal),
        out);
    verify(repo.div_u, out);
}

TEST_F(div, flow_convention) {
    auto out = make_storage<cells>();
    run_single_stage(div_functor_flow_convention_connectivity(),
        backend_t(),
        make_grid(),
        make_storage<edges>(repo.u),
        make_storage<edges>(repo.edge_length),
        make_storage<cells>(repo.cell_area_reciprocal),
        out);
    verify(repo.div_u, out);
}
