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

#pragma once

#include <tools/verifier.hpp>
#include "curl_functors.hpp"
#include "div_functors.hpp"
#include "grad_functors.hpp"
#include "operators_repository.hpp"
#include "../benchmarker.hpp"

namespace ico_operators {

    using x_interval = axis< 1 >::full_interval;

    template < uint_t Color >
    struct lap_functor {
        typedef in_accessor< 0, icosahedral_topology_t::cells, extent< -1, 0, -1, 0 > > in_cells;
        typedef in_accessor< 1, icosahedral_topology_t::edges > dual_edge_length_reciprocal;
        typedef in_accessor< 2, icosahedral_topology_t::vertices, extent< 0, 1, 0, 1 > > in_vertices;
        typedef in_accessor< 3, icosahedral_topology_t::edges > edge_length_reciprocal;
        typedef inout_accessor< 4, icosahedral_topology_t::edges > out_edges;
        typedef boost::mpl::
            vector< in_cells, dual_edge_length_reciprocal, in_vertices, edge_length_reciprocal, out_edges > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            constexpr auto neighbors_offsets_cell = connectivity< edges, cells, Color >::offsets();

            float_type grad_n{(eval(in_cells(neighbors_offsets_cell[1])) - eval(in_cells(neighbors_offsets_cell[0]))) *
                              eval(dual_edge_length_reciprocal())};

            constexpr auto neighbors_offsets_vertex = connectivity< edges, vertices, Color >::offsets();
            float_type grad_tau{
                (eval(in_vertices(neighbors_offsets_vertex[1])) - eval(in_vertices(neighbors_offsets_vertex[0]))) *
                eval(edge_length_reciprocal())};

            eval(out_edges()) = grad_n - grad_tau;
        }
    };

    bool test_lap(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {
        repository repository(x, y, z);
        repository.init_fields();
        repository.generate_lap_ref();

        icosahedral_topology_t &icosahedral_grid = repository.icosahedral_grid();
        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        const uint_t halo_nc = repository.halo_nc;
        const uint_t halo_mc = repository.halo_mc;
        const uint_t halo_k = repository.halo_k;

        typedef gridtools::layout_map< 2, 1, 0 > layout_t;

        halo_descriptor di{halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        halo_descriptor dj{halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        auto grid_ = make_grid(icosahedral_grid, di, dj, d3);

        using edge_storage_type = repository::edge_storage_type;
        using vertex_storage_type = repository::vertex_storage_type;
        using cell_storage_type = repository::cell_storage_type;

        using edge_2d_storage_type = repository::edge_2d_storage_type;
        using cell_2d_storage_type = repository::cell_2d_storage_type;
        using vertex_2d_storage_type = repository::vertex_2d_storage_type;

        using vertices_4d_storage_type = repository::vertices_4d_storage_type;
        using cells_4d_storage_type = repository::cells_4d_storage_type;

        using edges_of_cells_storage_type = repository::edges_of_cells_storage_type;
        using edges_of_vertices_storage_type = repository::edges_of_vertices_storage_type;

        // for div
        auto &cell_area_reciprocal = repository.cell_area_reciprocal();
        auto &edge_length = repository.edge_length();
        // for div weights
        auto &orientation_of_normal = repository.orientation_of_normal();
        auto div_weights = icosahedral_grid.make_storage< icosahedral_topology_t::cells,
            float_type,
            typename repository::halo_5d_t,
            selector< 1, 1, 1, 1, 1 > >("weights", 3);
        // for curl
        auto &dual_area_reciprocal = repository.dual_area_reciprocal();
        auto &dual_edge_length = repository.dual_edge_length();
        // for curl weights
        auto &edge_orientation = repository.edge_orientation();
        vertices_4d_storage_type curl_weights(icosahedral_grid.make_storage< icosahedral_topology_t::vertices,
                                              float_type,
                                              typename repository::halo_5d_t,
                                              selector< 1, 1, 1, 1, 1 > >("curl_weights", 6));
        // for lap
        auto &dual_edge_length_reciprocal = repository.dual_edge_length_reciprocal();
        auto &edge_length_reciprocal = repository.edge_length_reciprocal();

        auto &in_edges = repository.u();
        auto out_edges =
            icosahedral_grid.make_storage< icosahedral_topology_t::edges, float_type, typename repository::halo_t >(
                "out");
        auto &ref_edges = repository.lap_ref();

        bool result = true;
#if FLOAT_PRECISION == 4
        verifier ver(1e-6);
#else
        verifier ver(1e-9);
#endif
        array< array< uint_t, 2 >, 4 > halos = {
            {{halo_nc + 1, halo_nc + 1}, {0, 0}, {halo_mc + 1, halo_mc + 1}, {halo_k, halo_k}}};

        /*
         * prep stencil for div and curl weights
         */
        {
            // div
            typedef arg< 0, edge_2d_storage_type, enumtype::edges > p_edge_length;
            typedef arg< 1, cell_2d_storage_type, enumtype::cells > p_cell_area_reciprocal;
            typedef arg< 2, edges_of_cells_storage_type, enumtype::cells > p_orientation_of_normal;
            typedef arg< 3, cells_4d_storage_type, enumtype::cells > p_div_weights;

            // curl
            typedef arg< 4, vertex_2d_storage_type, enumtype::vertices > p_dual_area_reciprocal;
            typedef arg< 5, edge_2d_storage_type, enumtype::edges > p_dual_edge_length;
            typedef arg< 6, vertices_4d_storage_type, enumtype::vertices > p_curl_weights;
            typedef arg< 7, edges_of_vertices_storage_type, enumtype::vertices > p_edge_orientation;

            auto stencil_ = gridtools::make_computation< backend_t >(
                grid_,
                p_edge_length{} = edge_length,
                p_cell_area_reciprocal{} = cell_area_reciprocal,
                p_orientation_of_normal{} = orientation_of_normal,
                p_div_weights{} = div_weights,
                p_dual_area_reciprocal{} = dual_area_reciprocal,
                p_dual_edge_length{} = dual_edge_length,
                p_curl_weights{} = curl_weights,
                p_edge_orientation{} = edge_orientation,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_prep_functor, icosahedral_topology_t, icosahedral_topology_t::cells >(
                        p_edge_length(), p_cell_area_reciprocal(), p_orientation_of_normal(), p_div_weights()),
                    gridtools::make_stage< curl_prep_functor,
                        icosahedral_topology_t,
                        icosahedral_topology_t::vertices >(
                        p_dual_area_reciprocal(), p_dual_edge_length(), p_curl_weights(), p_edge_orientation())));
            stencil_.run();
            stencil_.sync_bound_data_stores();
        }

        /*
         * lap stencil weights
         */
        {
            // input
            typedef arg< 0, edge_storage_type, enumtype::edges > p_in_edges;

            // fields for div
            typedef arg< 1, cells_4d_storage_type, enumtype::cells > p_div_weights;
            typedef tmp_arg< 2, cell_storage_type, enumtype::cells > p_div_on_cells;

            // fields for curl
            typedef arg< 3, vertices_4d_storage_type, enumtype::vertices > p_curl_weights;
            typedef tmp_arg< 4, vertex_storage_type, enumtype::vertices > p_curl_on_vertices;

            // fields for lap
            typedef arg< 5, edge_2d_storage_type, enumtype::edges > p_dual_edge_length_reciprocal;
            typedef arg< 6, edge_2d_storage_type, enumtype::edges > p_edge_length_reciprocal;

            // output
            typedef arg< 7, edge_storage_type, enumtype::edges > p_out_edges;

            auto stencil_ = gridtools::make_computation< backend_t >(
                grid_,
                p_in_edges{} = in_edges,
                p_div_weights{} = div_weights,
                p_curl_weights{} = curl_weights,
                p_dual_edge_length_reciprocal{} = dual_edge_length_reciprocal,
                p_edge_length_reciprocal{} = edge_length_reciprocal,
                p_out_edges{} = out_edges,
                gridtools::make_multistage(
                    execute< forward >(),
                    make_stage< div_functor_reduction_into_scalar,
                        icosahedral_topology_t,
                        icosahedral_topology_t::cells >(p_in_edges(), p_div_weights(), p_div_on_cells()),
                    make_stage< curl_functor_weights, icosahedral_topology_t, icosahedral_topology_t::vertices >(
                        p_in_edges(), p_curl_weights(), p_curl_on_vertices()),
                    make_stage< lap_functor, icosahedral_topology_t, icosahedral_topology_t::edges >(p_div_on_cells(),
                        p_dual_edge_length_reciprocal(),
                        p_curl_on_vertices(),
                        p_edge_length_reciprocal(),
                        p_out_edges())));

            stencil_.run();

            in_edges.sync();
            dual_edge_length_reciprocal.sync();
            edge_length_reciprocal.sync();
            out_edges.sync();

            result = ver.verify(grid_, ref_edges, out_edges, halos) && result;

#ifdef BENCHMARK
            std::cout << "lap weights: ";
            benchmarker::run(stencil_, t_steps);
#endif
        }

        /*
         * lap stencil flow convention
         */
        {
            // input
            typedef arg< 0, edge_storage_type, enumtype::edges > p_in_edges;

            // fields for div
            typedef arg< 1, edge_2d_storage_type, enumtype::edges > p_edge_length;
            typedef arg< 2, cell_2d_storage_type, enumtype::cells > p_cell_area_reciprocal;
            typedef tmp_arg< 3, cell_storage_type, enumtype::cells > p_div_on_cells;

            // fields for curl
            typedef arg< 4, vertex_2d_storage_type, enumtype::vertices > p_dual_area_reciprocal;
            typedef arg< 5, edge_2d_storage_type, enumtype::edges > p_dual_edge_length;
            typedef tmp_arg< 6, vertex_storage_type, enumtype::vertices > p_curl_on_vertices;

            // fields for lap
            typedef arg< 7, edge_2d_storage_type, enumtype::edges > p_dual_edge_length_reciprocal;
            typedef arg< 8, edge_2d_storage_type, enumtype::edges > p_edge_length_reciprocal;

            // output
            typedef arg< 9, edge_storage_type, enumtype::edges > p_out_edges;

            auto stencil_ = gridtools::make_computation< backend_t >(
                grid_,
                p_in_edges{} = in_edges,
                p_edge_length{} = edge_length,
                p_cell_area_reciprocal{} = cell_area_reciprocal,
                p_dual_area_reciprocal{} = dual_area_reciprocal,
                p_dual_edge_length{} = dual_edge_length,
                p_dual_edge_length_reciprocal{} = dual_edge_length_reciprocal,
                p_edge_length_reciprocal{} = edge_length_reciprocal,
                p_out_edges{} = out_edges,
                gridtools::make_multistage(
                    execute< forward >(),
                    define_caches(cache< IJ, cache_io_policy::local >(p_div_on_cells())), // p_curl_on_vertices())),
                    make_stage< div_functor_flow_convention_connectivity,
                        icosahedral_topology_t,
                        icosahedral_topology_t::cells >(
                        p_in_edges(), p_edge_length(), p_cell_area_reciprocal(), p_div_on_cells()),
                    make_stage< curl_functor_flow_convention,
                        icosahedral_topology_t,
                        icosahedral_topology_t::vertices >(
                        p_in_edges(), p_dual_area_reciprocal(), p_dual_edge_length(), p_curl_on_vertices()),
                    make_stage< lap_functor, icosahedral_topology_t, icosahedral_topology_t::edges >(p_div_on_cells(),
                        p_dual_edge_length_reciprocal(),
                        p_curl_on_vertices(),
                        p_edge_length_reciprocal(),
                        p_out_edges())));

            stencil_.run();

            in_edges.sync();
            out_edges.sync();

            result = ver.verify(grid_, ref_edges, out_edges, halos) && result;

#ifdef BENCHMARK
            std::cout << "lap flow convention: ";
            benchmarker::run(stencil_, t_steps);
#endif
        }

        return result;
    }
}
