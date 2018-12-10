/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
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
        make_multistage(enumtype::execute<enumtype::forward>(),
            make_stage<div_prep_functor, topology_t, cells>(
                p_edge_length, p_cell_area_reciprocal, p_orientation_of_normal, p_div_weights),
            make_stage<div_functor_reduction_into_scalar, topology_t, cells>(p_in_edges, p_div_weights, p_out_cells)))
        .run();
}

TEST_F(div, flow_convention) {
    make_computation(p_in_edges = make_storage<edges>(repo.u),
        p_edge_length = make_storage<edges, edge_2d_storage_type>(repo.edge_length),
        p_cell_area_reciprocal = make_storage<cells, cell_2d_storage_type>(repo.cell_area_reciprocal),
        p_out_cells = out_cells,
        make_multistage(enumtype::execute<enumtype::forward>(),
            make_stage<div_functor_flow_convention_connectivity, topology_t, cells>(
                p_in_edges, p_edge_length, p_cell_area_reciprocal, p_out_cells)))
        .run();
}
