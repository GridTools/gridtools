#pragma once
/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#pragma once
#include <common/defs.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "operator_defs.hpp"

namespace ico_operators {

    using namespace gridtools;
    using namespace enumtype;
    using namespace expressions;

    template < uint_t Color >
    struct curl_prep_functor {
        typedef in_accessor< 0, icosahedral_topology_t::vertexes > dual_area_reciprocal;
        typedef in_accessor< 1, icosahedral_topology_t::edges, extent<-1,0,-1,0 > > dual_edge_length;
        typedef inout_accessor< 2, icosahedral_topology_t::vertexes, 5 > weights;
        typedef in_accessor< 3, icosahedral_topology_t::vertexes, extent<0,0,0,0>, 5 > edge_orientation;
        typedef boost::mpl::vector< dual_area_reciprocal, dual_edge_length, weights, edge_orientation > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            using edge_of_vertex_dim = dimension< 5 >;
            edge_of_vertex_dim::Index edge;

            constexpr auto neighbors_offsets = connectivity< vertexes, edges, Color >::offsets();
            ushort_t e = 0;
            for (auto neighbor_offset : neighbors_offsets) {
                eval(weights(edge + e)) += eval(edge_orientation(edge + e)) * eval(dual_edge_length(neighbor_offset)) *
                                           eval(dual_area_reciprocal());
                e++;
            }
        }
    };

    template < uint_t Color >
    struct curl_functor_weights {
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent<-1,0,-1,0> > in_edges;
        typedef in_accessor< 1, icosahedral_topology_t::vertexes, extent<0,0,0,0 >, 5 > weights;
        typedef inout_accessor< 2, icosahedral_topology_t::vertexes > out_vertexes;
        typedef boost::mpl::vector< in_edges, weights, out_vertexes > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            using edge_of_vertex_dim = dimension< 5 >;
            edge_of_vertex_dim::Index edge;

            double t{0.};
            constexpr auto neighbors_offsets = connectivity< vertexes, edges, Color >::offsets();
            ushort_t e = 0;
            for (auto neighbor_offset : neighbors_offsets) {
                t += eval(in_edges(neighbor_offset)) * eval(weights(edge + e));
                e++;
            }
            eval(out_vertexes()) = t;
        }
    };

    template < uint_t Color >
    struct curl_functor_flow_convention {
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 1 > > in_edges;
        typedef in_accessor< 1, icosahedral_topology_t::vertexes > dual_area_reciprocal;
        typedef in_accessor< 2, icosahedral_topology_t::edges, extent< 1 > > dual_edge_length;
        typedef inout_accessor< 3, icosahedral_topology_t::vertexes > out_vertexes;
        typedef boost::mpl::vector< in_edges, dual_area_reciprocal, dual_edge_length, out_vertexes > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            constexpr auto neighbor_offsets = connectivity< vertexes, edges, Color >::offsets();
            eval(out_vertexes()) = -eval(in_edges(neighbor_offsets[0])) * eval(dual_edge_length(neighbor_offsets[0])) +
                                   eval(in_edges(neighbor_offsets[1])) * eval(dual_edge_length(neighbor_offsets[1])) -
                                   eval(in_edges(neighbor_offsets[2])) * eval(dual_edge_length(neighbor_offsets[2])) +
                                   eval(in_edges(neighbor_offsets[3])) * eval(dual_edge_length(neighbor_offsets[3])) -
                                   eval(in_edges(neighbor_offsets[4])) * eval(dual_edge_length(neighbor_offsets[4])) +
                                   eval(in_edges(neighbor_offsets[5])) * eval(dual_edge_length(neighbor_offsets[5]));

            eval(out_vertexes()) *= eval(dual_area_reciprocal());
        }
    };
}
