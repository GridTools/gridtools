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

#include "curl_functors.hpp"
#include "div_functors.hpp"
#include "operators_repository.hpp"

using namespace gridtools;
using namespace ico_operators;

struct lap_functor {
    typedef in_accessor<0, enumtype::cells, extent<-1, 0, -1, 0>> in_cells;
    typedef in_accessor<1, enumtype::edges> dual_edge_length_reciprocal;
    typedef in_accessor<2, enumtype::vertices, extent<0, 1, 0, 1>> in_vertices;
    typedef in_accessor<3, enumtype::edges> edge_length_reciprocal;
    typedef inout_accessor<4, enumtype::edges> out_edges;
    using param_list =
        make_param_list<in_cells, dual_edge_length_reciprocal, in_vertices, edge_length_reciprocal, out_edges>;
    using location = enumtype::edges;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        constexpr auto neighbors_offsets_cell =
            connectivity<enumtype::edges, enumtype::cells, Evaluation::color>::offsets();

        float_type grad_n{(eval(in_cells(neighbors_offsets_cell[1])) - eval(in_cells(neighbors_offsets_cell[0]))) *
                          eval(dual_edge_length_reciprocal())};

        constexpr auto neighbors_offsets_vertex =
            connectivity<enumtype::edges, enumtype::vertices, Evaluation::color>::offsets();
        float_type grad_tau{
            (eval(in_vertices(neighbors_offsets_vertex[1])) - eval(in_vertices(neighbors_offsets_vertex[0]))) *
            eval(edge_length_reciprocal())};

        eval(out_edges()) = grad_n - grad_tau;
    }
};

struct lap : regression_fixture<2> {
    operators_repository repo = {d1(), d2()};

    storage_type<edges> out_edges = make_storage<edges>();

    arg<0, edges> p_in_edges;
    arg<1, edges> p_out_edges;
    arg<2, edges> p_edge_length;
    arg<3, edges> p_edge_length_reciprocal;
    arg<4, edges> p_dual_edge_length;
    arg<5, edges> p_dual_edge_length_reciprocal;
    arg<2, cells> p_cell_area_reciprocal;
    arg<3, vertices> p_dual_area_reciprocal;

    tmp_arg<0, cells> p_div_on_cells;
    tmp_arg<1, vertices> p_curl_on_vertices;

    ~lap() { verify(make_storage<edges>(repo.lap), out_edges); }
};

TEST_F(lap, weights) {
    using edges_of_cells_storage_type = storage_type_4d<cells, selector<1, 1, 1, 0, 1>>;
    using edges_of_vertices_storage_type = storage_type_4d<vertices, selector<1, 1, 1, 0, 1>>;

    arg<10, cells> p_orientation_of_normal;
    arg<11, cells> p_div_weights;
    arg<12, vertices> p_curl_weights;
    arg<13, vertices> p_edge_orientation;

    compute(p_edge_length = make_storage<edges, edge_2d_storage_type>(repo.edge_length),
        p_cell_area_reciprocal = make_storage<cells, cell_2d_storage_type>(repo.cell_area_reciprocal),
        p_orientation_of_normal = make_storage_4d<cells, edges_of_cells_storage_type>(3, repo.orientation_of_normal),
        p_div_weights = make_storage_4d<cells>(3),
        p_dual_area_reciprocal = make_storage<vertices, vertex_2d_storage_type>(repo.dual_area_reciprocal),
        p_dual_edge_length = make_storage<edges, edge_2d_storage_type>(repo.dual_edge_length),
        p_curl_weights = make_storage_4d<vertices>(6),
        p_edge_orientation = make_storage_4d<vertices, edges_of_vertices_storage_type>(6, repo.edge_orientation),
        p_in_edges = make_storage<edges>(repo.u),
        p_dual_edge_length_reciprocal = make_storage<edges, edge_2d_storage_type>(repo.dual_edge_length_reciprocal),
        p_edge_length_reciprocal = make_storage<edges, edge_2d_storage_type>(repo.edge_length_reciprocal),
        p_out_edges = out_edges,
        make_multistage(execute::forward(),
            make_stage<div_prep_functor>(p_edge_length, p_cell_area_reciprocal, p_orientation_of_normal, p_div_weights),
            make_stage<curl_prep_functor>(
                p_dual_area_reciprocal, p_dual_edge_length, p_curl_weights, p_edge_orientation),
            make_stage<div_functor_reduction_into_scalar>(p_in_edges, p_div_weights, p_div_on_cells),
            make_stage<curl_functor_weights>(p_in_edges, p_curl_weights, p_curl_on_vertices),
            make_stage<lap_functor>(p_div_on_cells,
                p_dual_edge_length_reciprocal,
                p_curl_on_vertices,
                p_edge_length_reciprocal,
                p_out_edges)));
}

TEST_F(lap, flow_convention) {
    compute(p_in_edges = make_storage<edges>(repo.u),
        p_edge_length = make_storage<edges, edge_2d_storage_type>(repo.edge_length),
        p_cell_area_reciprocal = make_storage<cells, cell_2d_storage_type>(repo.cell_area_reciprocal),
        p_dual_area_reciprocal = make_storage<vertices, vertex_2d_storage_type>(repo.dual_area_reciprocal),
        p_dual_edge_length = make_storage<edges, edge_2d_storage_type>(repo.dual_edge_length),
        p_dual_edge_length_reciprocal = make_storage<edges, edge_2d_storage_type>(repo.dual_edge_length_reciprocal),
        p_edge_length_reciprocal = make_storage<edges, edge_2d_storage_type>(repo.edge_length_reciprocal),
        p_out_edges = out_edges,
        make_multistage(execute::forward(),
            define_caches(cache<cache_type::ij>(p_div_on_cells)),
            make_stage<div_functor_flow_convention_connectivity>(
                p_in_edges, p_edge_length, p_cell_area_reciprocal, p_div_on_cells),
            make_stage<curl_functor_flow_convention>(
                p_in_edges, p_dual_area_reciprocal, p_dual_edge_length, p_curl_on_vertices),
            make_stage<lap_functor>(p_div_on_cells,
                p_dual_edge_length_reciprocal,
                p_curl_on_vertices,
                p_edge_length_reciprocal,
                p_out_edges)));
}
