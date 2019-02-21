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

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "curl_functors.hpp"
#include "operators_repository.hpp"

using namespace gridtools;
using namespace ico_operators;

struct curl : regression_fixture<2> {
    operators_repository repo = {d1(), d2()};

    arg<0, edges> p_in_edges;
    arg<1, vertices, vertex_2d_storage_type> p_dual_area_reciprocal;
    arg<2, edges, edge_2d_storage_type> p_dual_edge_length;
    arg<3, vertices> p_out_vertices;

    storage_type<vertices> out_vertices = make_storage<vertices>();

    ~curl() {
        constexpr double precision = GT_FLOAT_PRECISION == 4 ? 1e-4 : 1e-9;
        verify(make_storage<vertices>(repo.curl_u), out_vertices, precision);
    }
};

TEST_F(curl, weights) {
    using edges_of_vertices_storage_type = storage_type_4d<vertices, selector<1, 1, 1, 0, 1>>;

    arg<10, vertices, storage_type_4d<vertices>> p_curl_weights;
    arg<11, vertices, edges_of_vertices_storage_type> p_edge_orientation;

    make_computation(p_dual_area_reciprocal = make_storage<vertices, vertex_2d_storage_type>(repo.dual_area_reciprocal),
        p_dual_edge_length = make_storage<edges, edge_2d_storage_type>(repo.dual_edge_length),
        p_curl_weights = make_storage_4d<vertices>(6),
        p_edge_orientation = make_storage_4d<vertices, edges_of_vertices_storage_type>(6, repo.edge_orientation),
        p_in_edges = make_storage<edges>(repo.u),
        p_out_vertices = out_vertices,
        make_multistage(execute::forward(),
            make_stage<curl_prep_functor, topology_t, vertices>(
                p_dual_area_reciprocal, p_dual_edge_length, p_curl_weights, p_edge_orientation),
            make_stage<curl_functor_weights, topology_t, vertices>(p_in_edges, p_curl_weights, p_out_vertices)))
        .run();
}

TEST_F(curl, flow_convention) {
    make_computation(p_in_edges = make_storage<edges>(repo.u),
        p_dual_area_reciprocal = make_storage<vertices, vertex_2d_storage_type>(repo.dual_area_reciprocal),
        p_dual_edge_length = make_storage<edges, edge_2d_storage_type>(repo.dual_edge_length),
        p_out_vertices = out_vertices,
        make_multistage(execute::parallel(),
            make_stage<curl_functor_flow_convention, topology_t, vertices>(
                p_in_edges, p_dual_area_reciprocal, p_dual_edge_length, p_out_vertices)))
        .run();
}
