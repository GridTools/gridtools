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
    struct div_prep_functor {
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 0, 1, 0, 1 > > edge_length;
        typedef in_accessor< 1, icosahedral_topology_t::cells > cell_area_reciprocal;
        typedef in_accessor< 2, icosahedral_topology_t::cells, extent< 0 >, 5 > orientation_of_normal;
        typedef inout_accessor< 3, icosahedral_topology_t::cells, 5 > weights;

        typedef boost::mpl::vector< edge_length, cell_area_reciprocal, orientation_of_normal, weights > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            using edge_of_cell_dim = dimension< 5 >;
            edge_of_cell_dim edge;

            constexpr auto neighbors_offsets = connectivity< cells, edges, Color >::offsets();
            ushort_t e = 0;
            for (auto neighbor_offset : neighbors_offsets) {
                eval(weights(edge + e)) = eval(orientation_of_normal(edge + e)) * eval(edge_length(neighbor_offset)) *
                                          eval(cell_area_reciprocal());
                e++;
            }
        }
    };

    template < uint_t Color >
    struct div_prep_functor_on_edges {
        typedef in_accessor< 0, icosahedral_topology_t::edges > edge_length;
        typedef in_accessor< 1, icosahedral_topology_t::cells, extent< -1, 0, -1, 0 > > cell_area_reciprocal;
        typedef inout_accessor< 2, icosahedral_topology_t::edges, 5 > l_over_A;

        typedef boost::mpl::vector< edge_length, cell_area_reciprocal, l_over_A > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            constexpr auto neighbors_offsets = connectivity< edges, cells, Color >::offsets();

            using cell_of_edge_dim = dimension< 5 >;
            cell_of_edge_dim cell;

            eval(l_over_A(cell + 0)) = eval(cell_area_reciprocal(neighbors_offsets[0])) * eval(edge_length());
            eval(l_over_A(cell + 1)) = eval(cell_area_reciprocal(neighbors_offsets[1])) * eval(edge_length());
        }
    };

    template < uint_t Color >
    struct div_functor {
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 0, 1, 0, 1 > > in_edges;
        typedef in_accessor< 1, icosahedral_topology_t::cells, extent< 0 >, 5 > weights;
        typedef inout_accessor< 2, icosahedral_topology_t::cells > out_cells;
        typedef boost::mpl::vector< in_edges, weights, out_cells > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            using edge_of_cells_dim = dimension< 5 >;
            edge_of_cells_dim edge;

            eval(out_cells()) = 0.;
            constexpr auto neighbors_offsets = connectivity< cells, edges, Color >::offsets();
            ushort_t e = 0;
            for (auto neighbor_offset : neighbors_offsets) {
                eval(out_cells()) += eval(in_edges(neighbor_offset)) * eval(weights(edge + e));
                e++;
            }
        }
    };

    template < uint_t Color >
    struct div_functor_reduction_into_scalar {
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 0, 1, 0, 1 > > in_edges;
        typedef in_accessor< 1, icosahedral_topology_t::cells, extent< 0 >, 5 > weights;
        typedef inout_accessor< 2, icosahedral_topology_t::cells > out_cells;
        typedef boost::mpl::vector< in_edges, weights, out_cells > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            using edge_of_cells_dim = dimension< 5 >;
            edge_of_cells_dim edge;

            double t{0.};
            constexpr auto neighbors_offsets = connectivity< cells, edges, Color >::offsets();
            ushort_t e = 0;
            for (auto neighbor_offset : neighbors_offsets) {
                t += eval(in_edges(neighbor_offset)) * eval(weights(edge + e));
                e++;
            }
            eval(out_cells()) = t;
        }
    };

    template < uint_t Color >
    struct div_functor_flow_convention {
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 0, 1, 0, 1 > > in_edges;
        typedef in_accessor< 1, icosahedral_topology_t::edges, extent< 0, 1, 0, 1 > > edge_length;
        typedef in_accessor< 2, icosahedral_topology_t::cells, extent< 0 > > cell_area_reciprocal;
        typedef inout_accessor< 3, icosahedral_topology_t::cells > out_cells;
        typedef boost::mpl::vector< in_edges, edge_length, cell_area_reciprocal, out_cells > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            auto ff = [](
                const double _in1, const double _in2, const double _res) -> double { return _in1 * _in2 + _res; };

            if (Color == 0)
                eval(out_cells()) = eval(on_edges(ff, 0.0, in_edges(), edge_length())) * eval(cell_area_reciprocal());
            else
                eval(out_cells()) = -eval(on_edges(ff, 0.0, in_edges(), edge_length())) * eval(cell_area_reciprocal());
        }
    };

    template < uint_t Color >
    struct div_functor_flow_convention_connectivity {
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 0, 1, 0, 1 > > in_edges;
        typedef in_accessor< 1, icosahedral_topology_t::edges, extent< 0, 1, 0, 1 > > edge_length;
        typedef in_accessor< 2, icosahedral_topology_t::cells > cell_area_reciprocal;
        typedef inout_accessor< 3, icosahedral_topology_t::cells > out_cells;
        typedef boost::mpl::vector< in_edges, edge_length, cell_area_reciprocal, out_cells > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            auto ff = [](
                const double _in1, const double _in2, const double _res) -> double { return _in1 * _in2 + _res; };

            double t{0.};
            constexpr auto neighbors_offsets = connectivity< cells, edges, Color >::offsets();
            for (auto neighbor_offset : neighbors_offsets) {
                t += eval(in_edges(neighbor_offset)) * eval(edge_length(neighbor_offset));
            }

            if (Color == 0)
                eval(out_cells()) = t * eval(cell_area_reciprocal());
            else
                eval(out_cells()) = -t * eval(cell_area_reciprocal());
        }
    };

    template < uint_t Color >
    struct div_functor_over_edges {
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 0, 1, 0, 1 > > in_edges;
        typedef in_accessor< 1, icosahedral_topology_t::edges, extent< 0, 1, 0, 1 > > edge_length;
        typedef inout_accessor< 2, icosahedral_topology_t::cells > out_cells;
        typedef boost::mpl::vector< in_edges, edge_length, out_cells > arg_list;

        template < typename Evaluation >
#ifdef __CUDACC__
        __device__
#else
        GT_FUNCTION
#endif
            static void
            Do(Evaluation &eval, x_interval) {
            constexpr auto neighbors_offsets = connectivity< edges, cells, Color >::offsets();

            double t{eval(in_edges()) * eval(edge_length())};

            if (Color == 0) {
                eval(out_cells(neighbors_offsets[0])) = -t;
                eval(out_cells(neighbors_offsets[1])) = t;
            } else {
                eval(out_cells(neighbors_offsets[0])) -= t;
                eval(out_cells(neighbors_offsets[1])) += t;
            }

#ifdef __CUDACC__
            __syncthreads();
#endif
        }
    };
    template < uint_t Color >
    struct divide_by_field {
        typedef in_accessor< 0, icosahedral_topology_t::cells, extent< -1, 0, -1, 0 > > cell_area_reciprocal;
        // library protects out accessors with extent
        //        typedef accessor< 1, enumtype::inout, icosahedral_topology_t::cells, extent< -1, 0, -1, 0 > >
        //        out_cells;
        typedef inout_accessor< 1, icosahedral_topology_t::cells > out_cells;

        typedef boost::mpl::vector< cell_area_reciprocal, out_cells > arg_list;
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            constexpr auto neighbors_offsets = connectivity< edges, cells, Color >::offsets();

            eval(out_cells()) *= eval(cell_area_reciprocal());
            if (Color == 0) {
                eval(out_cells(neighbors_offsets[0])) *= eval(cell_area_reciprocal(neighbors_offsets[0]));
                eval(out_cells(neighbors_offsets[1])) *= eval(cell_area_reciprocal(neighbors_offsets[1]));
            }
        }
    };

    using namespace gridtools;
    using namespace enumtype;
    using namespace expressions;

    template < uint_t Color >
    struct div_functor_over_edges_weights {
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 0 > > in_edges;
        typedef in_accessor< 1, icosahedral_topology_t::edges, extent< 0 >, 5 > l_over_A;
        // library protects out accessors with extent
        //        typedef accessor< 2, enumtype::inout, icosahedral_topology_t::cells, extent<-1,0,-1,0> > out_cells;
        typedef inout_accessor< 2, icosahedral_topology_t::cells > out_cells;
        typedef boost::mpl::vector< in_edges, l_over_A, out_cells > arg_list;

        template < typename Evaluation >
#ifdef __CUDACC__
        __device__
#else
        GT_FUNCTION
#endif
            static void
            Do(Evaluation &eval, x_interval) {
            constexpr auto neighbors_offsets = connectivity< edges, cells, Color >::offsets();

            using cell_of_edge_dim = dimension< 5 >;
            cell_of_edge_dim cell;

            if (Color == 0) {
                eval(out_cells()) = eval(in_edges());
                eval(out_cells(neighbors_offsets[0])) = -eval(in_edges()) * eval(l_over_A(cell + 0));
                eval(out_cells(neighbors_offsets[1])) = eval(in_edges()) * eval(l_over_A(cell + 1));
            } else {
                eval(out_cells(neighbors_offsets[0])) -= eval(in_edges()) * eval(l_over_A(cell + 0));
                eval(out_cells(neighbors_offsets[1])) += eval(in_edges()) * eval(l_over_A(cell + 1));
            }
#ifdef __CUDACC__
            __syncthreads();
#endif
        }
    };
}
