#pragma once
#include "operator_defs.hpp"

namespace ico_operators {

    using namespace gridtools;
    using namespace enumtype;
    using namespace expressions;

    using icosahedral_topology_t = repository::icosahedral_topology_t;

    template < uint_t Color >
    struct curl_prep_functor {
        typedef inout_accessor< 0, icosahedral_topology_t::vertexes > dual_area_reciprocal;
        typedef in_accessor< 1, icosahedral_topology_t::edges, extent< 1 > > dual_edge_length;
        typedef inout_accessor< 2, icosahedral_topology_t::vertexes, 5 > weights;
        typedef in_accessor< 3, icosahedral_topology_t::vertexes, extent< 1 >, 5 > edge_orientation;
        typedef boost::mpl::vector< dual_area_reciprocal, dual_edge_length, weights, edge_orientation > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            using edge_of_vertex_dim = dimension< 5 >;
            edge_of_vertex_dim::Index edge;

            constexpr auto neighbors_offsets = connectivity< vertexes, edges, Color >::offsets();
            ushort_t e = 0;
            for (auto neighbor_offset : neighbors_offsets) {
                eval(weights(edge + e)) += eval(edge_orientation(edge + e)) * eval(dual_edge_length(neighbor_offset)) /
                                           eval(dual_area_reciprocal());
                e++;
            }
        }
    };

    template < uint_t Color >
    struct curl_functor_weights {
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 1 > > in_edges;
        typedef in_accessor< 1, icosahedral_topology_t::vertexes, extent< 1 >, 5 > weights;
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
