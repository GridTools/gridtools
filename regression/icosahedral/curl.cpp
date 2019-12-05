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
#include <gridtools/tools/regression_fixture.hpp>

#include "curl_functors.hpp"
#include "operators_repository.hpp"

using namespace gridtools;
using namespace ico_operators;

struct curl : regression_fixture<2> {
    operators_repository repo = {d1(), d2()};
    static constexpr double precision = GT_FLOAT_PRECISION == 4 ? 1e-4 : 1e-9;
};

const double curl::precision;

TEST_F(curl, weights) {
    auto spec = [](auto reciprocal, auto edge_length, auto in_edges, auto out) {
        GT_DECLARE_ICO_TMP((array<float_type, 6>), vertices, weights);
        return execute_parallel()
            .ij_cached(weights)
            .stage(curl_prep_functor(), reciprocal, edge_length, weights)
            .stage(curl_functor_weights(), in_edges, weights, out);
    };
    auto out = make_storage<vertices>();
    run(spec,
        backend_t(),
        make_grid(),
        make_storage<vertices>(repo.dual_area_reciprocal),
        make_storage<edges>(repo.dual_edge_length),
        make_storage<edges>(repo.u),
        out);
    verify(repo.curl_u, out, precision);
}

TEST_F(curl, flow_convention) {
    auto out = make_storage<vertices>();
    easy_run(curl_functor_flow_convention(),
        backend_t(),
        make_grid(),
        make_storage<edges>(repo.u),
        make_storage<vertices>(repo.dual_area_reciprocal),
        make_storage<edges>(repo.dual_edge_length),
        out);
    verify(repo.curl_u, out, precision);
}
