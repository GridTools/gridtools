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
    struct grad_n {
        typedef in_accessor< 0, icosahedral_topology_t::cells, extent< 1 > > in_cells;
        typedef in_accessor< 1, icosahedral_topology_t::edges, extent< 1 > > dual_edge_length_reciprocal;
        typedef inout_accessor< 2, icosahedral_topology_t::edges > out_edges;
        typedef boost::mpl::vector< in_cells, dual_edge_length_reciprocal, out_edges > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            constexpr auto neighbors_offsets = connectivity< edges, cells, Color >::offsets();

            eval(out_edges()) = (eval(in_cells(neighbors_offsets[0])) - eval(in_cells(neighbors_offsets[1]))) *
                                eval(dual_edge_length_reciprocal());
        }
    };

    template < uint_t Color >
    struct grad_tau {
        typedef in_accessor< 0, icosahedral_topology_t::vertices, extent< 1 > > in_vertices;
        typedef in_accessor< 1, icosahedral_topology_t::edges, extent< 1 > > edge_length_reciprocal;
        typedef inout_accessor< 2, icosahedral_topology_t::edges > out_edges;
        typedef boost::mpl::vector< in_vertices, edge_length_reciprocal, out_edges > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            constexpr auto neighbors_offsets = connectivity< edges, vertices, Color >::offsets();
            eval(out_edges()) = (eval(in_vertices(neighbors_offsets[1])) - eval(in_vertices(neighbors_offsets[0]))) *
                                eval(edge_length_reciprocal());
        }
    };
}
