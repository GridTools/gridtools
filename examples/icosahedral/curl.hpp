#pragma once

#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "operators_repository.hpp"
#include "../benchmarker.hpp"
#include "operator_defs.hpp"
#include "curl_functors.hpp"

using namespace gridtools;
using namespace enumtype;

namespace ico_operators {

    bool test_curl(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        repository repo(x, y, z);
        repo.init_fields();
        repo.generate_curl_ref();

        icosahedral_topology_t &icosahedral_grid = repo.icosahedral_grid();
        uint_t d1 = repo.idim();
        uint_t d2 = repo.jdim();
        uint_t d3 = repo.kdim();

        const uint_t halo_nc = repo.halo_nc;
        const uint_t halo_mc = repo.halo_mc;
        const uint_t halo_k = repo.halo_k;

        typedef gridtools::layout_map< 2, 1, 0 > layout_t;

        using edge_storage_type = repository::edge_storage_type;
        using cell_storage_type = repository::cell_storage_type;
        using vertex_storage_type = repository::vertex_storage_type;
        using vertex_2d_storage_type = repository::vertex_2d_storage_type;
        using edge_2d_storage_type = repository::edge_2d_storage_type;

        using vertices_4d_storage_type = repository::vertices_4d_storage_type;
        using edges_of_vertices_storage_type = repository::edges_of_vertices_storage_type;

        auto &in_edges = repo.u();
        auto &dual_area_reciprocal = repo.dual_area_reciprocal();
        auto &dual_edge_length = repo.dual_edge_length();
        auto &ref_vertices = repo.curl_u_ref();
        auto &out_vertices = repo.out_vertex();

        vertices_4d_storage_type curl_weights(
            icosahedral_grid.make_storage< icosahedral_topology_t::vertices, double, selector< 1, 1, 1, 1, 1 > >(
                "weights", 6));
        edges_of_vertices_storage_type &edge_orientation = repo.edge_orientation();

        out_vertices.initialize(0.0);
        curl_weights.initialize(0.0);

        array< uint_t, 5 > di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        array< uint_t, 5 > dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        gridtools::grid< axis, icosahedral_topology_t > grid_(icosahedral_grid, di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        bool result = true;

        {
            typedef arg< 0, vertex_2d_storage_type, enumtype::vertices > p_dual_area_reciprocal;
            typedef arg< 1, edge_2d_storage_type, enumtype::edges > p_dual_edge_length;
            typedef arg< 2, vertices_4d_storage_type, enumtype::vertices > p_curl_weights;
            typedef arg< 3, edges_of_vertices_storage_type, enumtype::vertices > p_edge_orientation;

            typedef boost::mpl::vector< p_dual_area_reciprocal, p_dual_edge_length, p_curl_weights, p_edge_orientation >
                accessor_list_t;

            gridtools::aggregator_type< accessor_list_t > domain(
                boost::fusion::make_vector(&dual_area_reciprocal, &dual_edge_length, &curl_weights, &edge_orientation));

            auto stencil_ = gridtools::make_computation< backend_t >(
                domain,
                grid_,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< curl_prep_functor,
                        icosahedral_topology_t,
                        icosahedral_topology_t::vertices >(
                        p_dual_area_reciprocal(), p_dual_edge_length(), p_curl_weights(), p_edge_orientation())));
            stencil_->ready();
            stencil_->steady();
            stencil_->run();

#ifdef __CUDACC__
            dual_area_reciprocal.d2h_update();
            dual_edge_length.d2h_update();
            curl_weights.d2h_update();
            edge_orientation.d2h_update();
#endif
        }

        {
            typedef arg< 0, edge_storage_type, enumtype::edges > p_in_edges;
            typedef arg< 1, vertices_4d_storage_type, enumtype::vertices > p_curl_weights;
            typedef arg< 2, vertex_storage_type, enumtype::vertices > p_out_vertices;

            typedef boost::mpl::vector< p_in_edges, p_curl_weights, p_out_vertices > accessor_list_t;

            gridtools::aggregator_type< accessor_list_t > domain(
                boost::fusion::make_vector(&in_edges, &curl_weights, &out_vertices));

            auto stencil_ = gridtools::make_computation< backend_t >(
                domain,
                grid_,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< curl_functor_weights,
                        icosahedral_topology_t,
                        icosahedral_topology_t::vertices >(p_in_edges(), p_curl_weights(), p_out_vertices())));

            stencil_->ready();
            stencil_->steady();
            stencil_->run();

#ifdef __CUDACC__
            in_edges.d2h_update();
            curl_weights.d2h_update();
            out_vertices.d2h_update();
#endif

#if FLOAT_PRECISION == 4
            verifier ver(1e-12);
#else
            verifier ver(1e-6);
#endif
            array< array< uint_t, 2 >, 4 > halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
            result = result && ver.verify(grid_, ref_vertices, out_vertices, halos);

#ifdef BENCHMARK
            std::cout << "curl weights  ";
            benchmarker::run(stencil_, t_steps);
#endif
        }

        {
            typedef arg< 0, edge_storage_type, enumtype::edges > p_in_edges;
            typedef arg< 1, vertex_2d_storage_type, enumtype::vertices > p_dual_area_reciprocal;
            typedef arg< 2, edge_2d_storage_type, enumtype::edges > p_dual_edge_length;
            typedef arg< 3, vertex_storage_type, enumtype::vertices > p_out_vertices;

            typedef boost::mpl::vector< p_in_edges, p_dual_area_reciprocal, p_dual_edge_length, p_out_vertices >
                accessor_list_t;

            gridtools::aggregator_type< accessor_list_t > domain(
                boost::fusion::make_vector(&in_edges, &dual_area_reciprocal, &dual_edge_length, &out_vertices));

            auto stencil_ = gridtools::make_computation< backend_t >(
                domain,
                grid_,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< curl_functor_flow_convention,
                        icosahedral_topology_t,
                        icosahedral_topology_t::vertices >(
                        p_in_edges(), p_dual_area_reciprocal(), p_dual_edge_length(), p_out_vertices())));

            stencil_->ready();
            stencil_->steady();
            stencil_->run();

#ifdef __CUDACC__
            in_edges.d2h_update();
            dual_area_reciprocal.d2h_update();
            dual_edge_length.d2h_update();
            out_vertices.d2h_update();
#endif

            verifier ver(1e-9);

            array< array< uint_t, 2 >, 4 > halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
            result = result && ver.verify(grid_, ref_vertices, out_vertices, halos);

#ifdef BENCHMARK
            std::cout << "curl flow convention  ";
            benchmarker::run(stencil_, t_steps);
#endif
        }

        return result;
    }
}
