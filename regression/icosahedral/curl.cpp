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
        constexpr double precision = FLOAT_PRECISION == 4 ? 1e-4 : 1e-9;
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
        make_multistage(execute<execution::forward>(),
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
        make_multistage(execute<execution::parallel>(),
            make_stage<curl_functor_flow_convention, topology_t, vertices>(
                p_in_edges, p_dual_area_reciprocal, p_dual_edge_length, p_out_vertices)))
        .run();
}
