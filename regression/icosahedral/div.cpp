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

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "div_functors.hpp"
#include "operators_repository.hpp"

using namespace gridtools;
using namespace ico_operators;

struct div : regression_fixture<2> {
    operators_repository repo = {d1(), d2()};

    storage_type<cells> out_cells = make_storage<cells>();

    arg<0, edges> p_in_edges;
    arg<1, edges, edge_2d_storage_type> p_edge_length;
    arg<2, cells, cell_2d_storage_type> p_cell_area_reciprocal;
    arg<3, cells> p_out_cells;

    ~div() { verify(make_storage<cells>(repo.div_u), out_cells); }
};

TEST_F(div, reduction_into_scalar) {
    using edges_of_cells_storage_type = storage_type_4d<cells, selector<1, 1, 1, 0, 1>>;

    arg<10, cells, storage_type_4d<cells>> p_div_weights;
    arg<11, cells, edges_of_cells_storage_type> p_orientation_of_normal;

    make_computation(p_in_edges = make_storage<edges>(repo.u),
        p_edge_length = make_storage<edges, edge_2d_storage_type>(repo.edge_length),
        p_cell_area_reciprocal = make_storage<cells, cell_2d_storage_type>(repo.cell_area_reciprocal),
        p_orientation_of_normal = make_storage_4d<cells, edges_of_cells_storage_type>(3, repo.orientation_of_normal),
        p_div_weights = make_storage_4d<cells>(3),
        p_out_cells = out_cells,
        make_multistage(execute::forward(),
            make_stage<div_prep_functor, cells>(
                p_edge_length, p_cell_area_reciprocal, p_orientation_of_normal, p_div_weights),
            make_stage<div_functor_reduction_into_scalar, cells>(p_in_edges, p_div_weights, p_out_cells)))
        .run();
}

TEST_F(div, flow_convention) {
    make_computation(p_in_edges = make_storage<edges>(repo.u),
        p_edge_length = make_storage<edges, edge_2d_storage_type>(repo.edge_length),
        p_cell_area_reciprocal = make_storage<cells, cell_2d_storage_type>(repo.cell_area_reciprocal),
        p_out_cells = out_cells,
        make_multistage(execute::forward(),
            make_stage<div_functor_flow_convention_connectivity, cells>(
                p_in_edges, p_edge_length, p_cell_area_reciprocal, p_out_cells)))
        .run();
}
