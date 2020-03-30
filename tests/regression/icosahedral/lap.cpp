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

#include "curl_functors.hpp"
#include "div_functors.hpp"
#include "operators_repository.hpp"

using namespace gridtools;
using namespace ico_operators;

struct lap_functor {
    typedef in_accessor<0, cells, extent<-1, 0, -1, 0>> in_cells;
    typedef in_accessor<1, edges> dual_edge_length_reciprocal;
    typedef in_accessor<2, vertices, extent<0, 1, 0, 1>> in_vertices;
    typedef in_accessor<3, edges> edge_length_reciprocal;
    typedef inout_accessor<4, edges> out_edges;
    using param_list =
        make_param_list<in_cells, dual_edge_length_reciprocal, in_vertices, edge_length_reciprocal, out_edges>;
    using location = edges;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        float_type grad_n = 0;
        int e = 0;
        eval.for_neighbors([&](auto in) { grad_n += (e++ ? 1 : -1) * in; }, in_cells());
        grad_n *= eval(dual_edge_length_reciprocal());

        float_type grad_tau = 0;
        e = 0;
        eval.for_neighbors([&](auto in) { grad_tau += (e++ ? 1 : -1) * in; }, in_vertices());
        grad_tau *= eval(edge_length_reciprocal());

        eval(out_edges()) = grad_n - grad_tau;
    }
};

struct lap : regression_fixture<2> {
    operators_repository repo = {d(0), d(1)};
};

TEST_F(lap, weights) {
    auto spec = [](auto edge_length,
                    auto cell_area_reciprocal,
                    auto dual_area_reciprocal,
                    auto dual_edge_length,
                    auto in_edges,
                    auto dual_edge_length_reciprocal,
                    auto edge_length_reciprocal,
                    auto out) {
        GT_DECLARE_ICO_TMP(float_type, cells, div_on_cells);
        GT_DECLARE_ICO_TMP(float_type, vertices, curl_on_vertices);
        GT_DECLARE_ICO_TMP((array<float_type, 3>), cells, div_weights);
        GT_DECLARE_ICO_TMP((array<float_type, 6>), vertices, curl_weights);
        // sorry, curl_weights doesn't fit the ij_cache on daint gpu :(
        return execute_parallel()
            .ij_cached(div_on_cells, curl_on_vertices, div_weights /*, curl_weights*/)
            .stage(div_prep_functor(), edge_length, cell_area_reciprocal, div_weights)
            .stage(curl_prep_functor(), dual_area_reciprocal, dual_edge_length, curl_weights)
            .stage(div_functor_reduction_into_scalar(), in_edges, div_weights, div_on_cells)
            .stage(curl_functor_weights(), in_edges, curl_weights, curl_on_vertices)
            .stage(lap_functor(),
                div_on_cells,
                dual_edge_length_reciprocal,
                curl_on_vertices,
                edge_length_reciprocal,
                out);
    };
    auto out = make_storage<edges>();
    run(spec,
        backend_t(),
        make_grid(),
        make_storage<edges>(repo.edge_length),
        make_storage<cells>(repo.cell_area_reciprocal),
        make_storage<vertices>(repo.dual_area_reciprocal),
        make_storage<edges>(repo.dual_edge_length),
        make_storage<edges>(repo.u),
        make_storage<edges>(repo.dual_edge_length_reciprocal),
        make_storage<edges>(repo.edge_length_reciprocal),
        out);
    verify(make_storage<edges>(repo.lap), out);
}

TEST_F(lap, flow_convention) {
    auto spec = [](auto in_edges,
                    auto edge_length,
                    auto cell_area_reciprocal,
                    auto dual_area_reciprocal,
                    auto dual_edge_length,
                    auto dual_edge_length_reciprocal,
                    auto edge_length_reciprocal,
                    auto out) {
        GT_DECLARE_ICO_TMP(float_type, cells, div_on_cells);
        GT_DECLARE_ICO_TMP(float_type, vertices, curl_on_vertices);
        return execute_parallel()
            .ij_cached(div_on_cells, curl_on_vertices)
            .stage(
                div_functor_flow_convention_connectivity(), in_edges, edge_length, cell_area_reciprocal, div_on_cells)
            .stage(curl_functor_flow_convention(), in_edges, dual_area_reciprocal, dual_edge_length, curl_on_vertices)
            .stage(lap_functor(),
                div_on_cells,
                dual_edge_length_reciprocal,
                curl_on_vertices,
                edge_length_reciprocal,
                out);
    };
    auto out = make_storage<edges>();
    run(spec,
        backend_t(),
        make_grid(),
        make_storage<edges>(repo.u),
        make_storage<edges>(repo.edge_length),
        make_storage<cells>(repo.cell_area_reciprocal),
        make_storage<vertices>(repo.dual_area_reciprocal),
        make_storage<edges>(repo.dual_edge_length),
        make_storage<edges>(repo.dual_edge_length_reciprocal),
        make_storage<edges>(repo.edge_length_reciprocal),
        out);
    verify(make_storage<edges>(repo.lap), out);
}
