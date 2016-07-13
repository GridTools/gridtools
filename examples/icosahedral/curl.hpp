//
// Created by Xiaolin Guo on 19.04.16.
//
#pragma once

#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "operators_repository.hpp"
#include "../benchmarker.hpp"

using namespace gridtools;
using namespace expressions;
using namespace enumtype;

namespace curlex {

    using backend_t = BACKEND;
    using icosahedral_topology_t = operators::repository::icosahedral_topology_t;


    typedef gridtools::interval< level<0, -1>, level<1, -1> > x_interval;
    typedef gridtools::interval< level<0, -2>, level<1, 1> > axis;

    template < uint_t Color >
    struct curl_prep_functor {
        typedef inout_accessor<0, icosahedral_topology_t::vertexes > dual_area;
        typedef in_accessor<1, icosahedral_topology_t::edges, extent<1> > dual_edge_length;
        typedef inout_accessor<2, icosahedral_topology_t::vertexes, 5 > weights;
        typedef in_accessor<3, icosahedral_topology_t::vertexes, extent<1>, 5 > edge_orientation;
        typedef boost::mpl::vector<dual_area, dual_edge_length, weights, edge_orientation> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            using edge_of_vertex_dim = dimension< 5 >;
            edge_of_vertex_dim::Index edge;

            constexpr auto neighbors_offsets = connectivity< vertexes, edges, Color >::offsets();
//            ushort_t e=0;
//            for (auto neighbor_offset : neighbors_offsets) {
//                eval(weights(edge + e)) += eval(edge_orientation(edge + e)) * eval(dual_edge_length(neighbor_offset)) / eval(dual_area());
//                e++;
//            }
            eval(dual_area()) = 1. / eval(dual_area());
        }
    };

    template < uint_t Color >
    struct curl_functor_weights {
        typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > in_edges;
        typedef in_accessor<1, icosahedral_topology_t::vertexes, extent<1>, 5 > weights;
        typedef inout_accessor<2, icosahedral_topology_t::vertexes > out_vertexes;
        typedef boost::mpl::vector<in_edges, weights, out_vertexes> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            using edge_of_vertex_dim = dimension< 5 >;
            edge_of_vertex_dim::Index edge;

            double t{0.};
            constexpr auto neighbors_offsets = connectivity< vertexes, edges, Color >::offsets();
            ushort_t e=0;
            for (auto neighbor_offset : neighbors_offsets) {
                t += eval(in_edges(neighbor_offset)) * eval(weights(edge+e));
                e++;
            }
            eval(out_vertexes()) = t;
        }
    };

    template < uint_t Color >
    struct curl_functor_flow_convention {
        typedef in_accessor<0, icosahedral_topology_t::edges, extent<1> > in_edges;
        typedef in_accessor<1, icosahedral_topology_t::vertexes, extent<1> > dual_area;
        typedef in_accessor<2, icosahedral_topology_t::edges, extent<1> > dual_edge_length;
        typedef inout_accessor<3, icosahedral_topology_t::vertexes > out_vertexes;
        typedef boost::mpl::vector<in_edges, dual_area, dual_edge_length, out_vertexes> arg_list;

        template<typename Evaluation>
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval)
        {
            constexpr auto neighbor_offsets = connectivity< vertexes, edges, Color >::offsets();
            eval(out_vertexes()) = - eval(in_edges(neighbor_offsets[0])) * eval(dual_edge_length(neighbor_offsets[0]))
                    + eval(in_edges(neighbor_offsets[1])) * eval(dual_edge_length(neighbor_offsets[1]))
                    - eval(in_edges(neighbor_offsets[2])) * eval(dual_edge_length(neighbor_offsets[2]))
                    + eval(in_edges(neighbor_offsets[3])) * eval(dual_edge_length(neighbor_offsets[3]))
                    - eval(in_edges(neighbor_offsets[4])) * eval(dual_edge_length(neighbor_offsets[4]))
                    + eval(in_edges(neighbor_offsets[5])) * eval(dual_edge_length(neighbor_offsets[5]));
            eval(out_vertexes()) *= eval(dual_area());
        }
    };

    bool test( uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify)
    {

        operators::repository repository(x, y, z);
        repository.init_fields();
        repository.generate_reference();

        icosahedral_topology_t& icosahedral_grid = repository.icosahedral_grid();
        uint_t d1 = repository.idim();
        uint_t d2 = repository.jdim();
        uint_t d3 = repository.kdim();

        const uint_t halo_nc = repository.halo_nc;
        const uint_t halo_mc = repository.halo_mc;
        const uint_t halo_k = repository.halo_k;

        typedef gridtools::layout_map<2, 1, 0> layout_t;

        using edge_storage_type = operators::repository::edge_storage_type;
        using cell_storage_type = operators::repository::cell_storage_type;
        using vertex_storage_type = operators::repository::vertex_storage_type;
        using edges_of_vertexes_storage_type = operators::repository::edges_of_vertexes_storage_type;

        auto& in_edges = repository.u();
        auto& dual_area = repository.dual_area();
        auto& dual_edge_length = repository.dual_edge_length();
        auto& ref_vertexes = repository.curl_u_ref();
        auto& weights_meta = repository.edges_of_vertexes_meta();
        auto out_vertexes = icosahedral_grid.make_storage<icosahedral_topology_t::vertexes, double>("out");

        edges_of_vertexes_storage_type curl_weights(weights_meta, "weights");
        edges_of_vertexes_storage_type &edge_orientation = repository.edge_orientation();


        out_vertexes.initialize(0.0);
        curl_weights.initialize(0.0);

        array<uint_t, 5> di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        array<uint_t, 5> dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        gridtools::grid<axis, icosahedral_topology_t> grid_(icosahedral_grid, di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        bool result = true;

        {
            typedef arg<0, vertex_storage_type> p_dual_area;
            typedef arg<1, edge_storage_type> p_dual_edge_length;
            typedef arg<2, edges_of_vertexes_storage_type> p_curl_weights;
            typedef arg<3, edges_of_vertexes_storage_type> p_edge_orientation;

            typedef boost::mpl::vector<p_dual_area, p_dual_edge_length, p_curl_weights, p_edge_orientation>
                    accessor_list_t;

            gridtools::aggregator_type<accessor_list_t> domain(
                    boost::fusion::make_vector(&dual_area, &dual_edge_length, &curl_weights, &edge_orientation));

            auto stencil_ = gridtools::make_computation<backend_t>(
                    domain,
                    grid_,
                    gridtools::make_multistage // mss_descriptor
                            (execute<forward>(),
                             gridtools::make_stage<curl_prep_functor, icosahedral_topology_t, icosahedral_topology_t::vertexes >(
                                     p_dual_area(), p_dual_edge_length(), p_curl_weights(), p_edge_orientation())
                            )
            );
            stencil_->ready();
            stencil_->steady();
            stencil_->run();

#ifdef __CUDACC__
            dual_area.d2h_update();
            dual_edge_length.d2h_update();
            curl_weights.d2h_update();
            edge_orientation.d2h_update();
#endif
        }
/*
        {
            typedef arg<0, edge_storage_type> p_in_edges;
            typedef arg<1, edges_of_vertexes_storage_type> p_curl_weights;
            typedef arg<2, vertex_storage_type> p_out_vertexes;

            typedef boost::mpl::vector<p_in_edges, p_curl_weights, p_out_vertexes>
                    accessor_list_t;

            gridtools::aggregator_type<accessor_list_t> domain(
                    boost::fusion::make_vector(&in_edges, &curl_weights, &out_vertexes));

            auto stencil_ = gridtools::make_computation<backend_t>(
                    domain,
                    grid_,
                    gridtools::make_multistage // mss_descriptor
                            (execute<forward>(),
                             gridtools::make_stage<curl_functor_weights, icosahedral_topology_t, icosahedral_topology_t::vertexes >(
                                     p_in_edges(), p_curl_weights(), p_out_vertexes())
                            )
            );

            stencil_->ready();
            stencil_->steady();
            stencil_->run();

#ifdef __CUDACC__
            in_edges.d2h_update();
            curl_weights.d2h_update();
            out_vertexes.d2h_update();
#endif

            verifier ver(1e-10);

            array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
            result = result && ver.verify(grid_, ref_vertexes, out_vertexes, halos);

#ifdef BENCHMARK
            std::cout << "curl weights  ";
            benchmarker::run(stencil_, t_steps);
#endif
        }
*/
        {
            typedef arg<0, edge_storage_type> p_in_edges;
            typedef arg<1, vertex_storage_type> p_dual_area;
            typedef arg<2, edge_storage_type> p_dual_edge_length;
            typedef arg<3, vertex_storage_type> p_out_vertexes;

            typedef boost::mpl::vector<p_in_edges, p_dual_area, p_dual_edge_length, p_out_vertexes>
                    accessor_list_t;

            gridtools::aggregator_type<accessor_list_t> domain(
                    boost::fusion::make_vector(&in_edges, &dual_area, &dual_edge_length, &out_vertexes));

            auto stencil_ = gridtools::make_computation<backend_t>(
                    domain,
                    grid_,
                    gridtools::make_multistage // mss_descriptor
                            (execute<forward>(),
                             gridtools::make_stage<curl_functor_flow_convention, icosahedral_topology_t, icosahedral_topology_t::vertexes >(
                                     p_in_edges(), p_dual_area(), p_dual_edge_length(), p_out_vertexes())
                            )
            );

            stencil_->ready();
            stencil_->steady();
            stencil_->run();

#ifdef __CUDACC__
            in_edges.d2h_update();
            dual_area.d2h_update();
            dual_edge_length.d2h_update();
            out_vertexes.d2h_update();
#endif

            verifier ver(1e-10);

            array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
            result = result && ver.verify(grid_, ref_vertexes, out_vertexes, halos);

#ifdef BENCHMARK
            std::cout << "curl flow convention  ";
            benchmarker::run(stencil_, t_steps);
#endif
        }

        return result;
    }
}
