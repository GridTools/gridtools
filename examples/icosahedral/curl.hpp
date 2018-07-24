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

#include "../benchmarker.hpp"
#include "curl_functors.hpp"
#include "operator_defs.hpp"
#include "operators_repository.hpp"
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/verifier.hpp>

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

        typedef gridtools::layout_map<2, 1, 0> layout_t;

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

        vertices_4d_storage_type curl_weights(icosahedral_grid.make_storage<icosahedral_topology_t::vertices,
                                              float_type,
                                              typename repository::halo_5d_t,
                                              selector<1, 1, 1, 1, 1>>("weights", 6));
        edges_of_vertices_storage_type &edge_orientation = repo.edge_orientation();

        curl_weights = vertices_4d_storage_type(*curl_weights.get_storage_info_ptr(), 0.0);

        halo_descriptor di{halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        halo_descriptor dj{halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        auto grid_ = make_grid(icosahedral_grid, di, dj, d3);

        bool result = true;

        {
            typedef arg<0, vertex_2d_storage_type, enumtype::vertices> p_dual_area_reciprocal;
            typedef arg<1, edge_2d_storage_type, enumtype::edges> p_dual_edge_length;
            typedef arg<2, vertices_4d_storage_type, enumtype::vertices> p_curl_weights;
            typedef arg<3, edges_of_vertices_storage_type, enumtype::vertices> p_edge_orientation;

            auto stencil_ = gridtools::make_computation<backend_t>(grid_,
                p_dual_area_reciprocal{} = dual_area_reciprocal,
                p_dual_edge_length{} = dual_edge_length,
                p_curl_weights{} = curl_weights,
                p_edge_orientation{} = edge_orientation,
                gridtools::make_multistage // mss_descriptor
                (execute<forward>(),
                    gridtools::make_stage<curl_prep_functor, icosahedral_topology_t, icosahedral_topology_t::vertices>(
                        p_dual_area_reciprocal(), p_dual_edge_length(), p_curl_weights(), p_edge_orientation())));
            stencil_.run();

            dual_area_reciprocal.sync();
            dual_edge_length.sync();
            curl_weights.sync();
            edge_orientation.sync();
        }

        {
            typedef arg<0, edge_storage_type, enumtype::edges> p_in_edges;
            typedef arg<1, vertices_4d_storage_type, enumtype::vertices> p_curl_weights;
            typedef arg<2, vertex_storage_type, enumtype::vertices> p_out_vertices;

            auto stencil_ = gridtools::make_computation<backend_t>(grid_,
                p_in_edges{} = in_edges,
                p_curl_weights{} = curl_weights,
                p_out_vertices{} = out_vertices,
                gridtools::make_multistage // mss_descriptor
                (execute<forward>(),
                    gridtools::make_stage<curl_functor_weights,
                        icosahedral_topology_t,
                        icosahedral_topology_t::vertices>(p_in_edges(), p_curl_weights(), p_out_vertices())));

            stencil_.run();

            in_edges.sync();
            curl_weights.sync();
            out_vertices.sync();

#if FLOAT_PRECISION == 4
            verifier ver(1e-4);
#else
            verifier ver(1e-9);
#endif
            array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
            result = result && ver.verify(grid_, ref_vertices, out_vertices, halos);

#ifdef BENCHMARK
            std::cout << "curl weights  ";
            benchmarker::run(stencil_, t_steps);
#endif
        }

        {
            typedef arg<0, edge_storage_type, enumtype::edges> p_in_edges;
            typedef arg<1, vertex_2d_storage_type, enumtype::vertices> p_dual_area_reciprocal;
            typedef arg<2, edge_2d_storage_type, enumtype::edges> p_dual_edge_length;
            typedef arg<3, vertex_storage_type, enumtype::vertices> p_out_vertices;

            auto stencil_ = gridtools::make_computation<backend_t>(grid_,
                p_in_edges{} = in_edges,
                p_dual_area_reciprocal{} = dual_area_reciprocal,
                p_dual_edge_length{} = dual_edge_length,
                p_out_vertices{} = out_vertices,
                gridtools::make_multistage // mss_descriptor
                (execute<parallel>(),
                    gridtools::make_stage<curl_functor_flow_convention,
                        icosahedral_topology_t,
                        icosahedral_topology_t::vertices>(
                        p_in_edges(), p_dual_area_reciprocal(), p_dual_edge_length(), p_out_vertices())));

            stencil_.run();

            in_edges.sync();
            dual_area_reciprocal.sync();
            dual_edge_length.sync();
            out_vertices.sync();

#if FLOAT_PRECISION == 4
            verifier ver(1e-4);
#else
            verifier ver(1e-9);
#endif

            array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
            result = result && ver.verify(grid_, ref_vertices, out_vertices, halos);

#ifdef BENCHMARK
            std::cout << "curl flow convention  ";
            benchmarker::run(stencil_, t_steps);
#endif
        }

        return result;
    }
} // namespace ico_operators
