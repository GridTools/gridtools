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
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "operators_repository.hpp"
#include "benchmarker.hpp"
#include "operator_defs.hpp"
#include "div_functors.hpp"
#include "backend_select.hpp"

using namespace gridtools;
using namespace enumtype;

namespace ico_operators {

    bool test_div(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        repository repo(x, y, z);
        repo.init_fields();
        repo.generate_div_ref();

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
        using cell_2d_storage_type = repository::cell_2d_storage_type;
        using vertex_storage_type = repository::vertex_storage_type;
        using vertex_2d_storage_type = repository::vertex_2d_storage_type;
        using edge_2d_storage_type = repository::edge_2d_storage_type;

        using vertices_4d_storage_type = repository::vertices_4d_storage_type;
        using cells_4d_storage_type = repository::cells_4d_storage_type;
        using edges_4d_storage_type = repository::edges_4d_storage_type;
        using edges_of_cells_storage_type = repository::edges_of_cells_storage_type;

        halo_descriptor di{halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        halo_descriptor dj{halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        auto grid_ = make_grid(icosahedral_grid, di, dj, d3);

#if FLOAT_PRECISION == 4
        verifier ver(1e-6);
#else
        verifier ver(1e-9);
#endif

        array< array< uint_t, 2 >, 4 > halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};

        auto &in_edges = repo.u();
        auto &cell_area_reciprocal = repo.cell_area_reciprocal();
        auto &orientation_of_normal = repo.orientation_of_normal();
        auto &edge_length = repo.edge_length();
        auto &ref_cells = repo.div_u_ref();
        auto out_cells =
            icosahedral_grid.make_storage< icosahedral_topology_t::cells, float_type, typename repository::halo_t >(
                "out");

        auto div_weights = icosahedral_grid.make_storage< icosahedral_topology_t::cells,
            float_type,
            typename repository::halo_5d_t,
            selector< 1, 1, 1, 1, 1 > >("weights", 3);

        auto l_over_A = icosahedral_grid.make_storage< icosahedral_topology_t::edges,
            float_type,
            typename repository::halo_5d_t,
            selector< 1, 1, 1, 1, 1 > >("l_over_A", 2);
        typedef decltype(out_cells) out_cells_storage;
        typedef decltype(l_over_A) l_over_A_storage;

        out_cells = out_cells_storage(*out_cells.get_storage_info_ptr(), 0.0);
        div_weights = cells_4d_storage_type(*div_weights.get_storage_info_ptr(), 0.0);
        l_over_A = l_over_A_storage(*l_over_A.get_storage_info_ptr(), 0.0);

        {
            typedef arg< 0, edge_2d_storage_type, enumtype::edges > p_edge_length;
            typedef arg< 1, cell_2d_storage_type, enumtype::cells > p_cell_area_reciprocal;
            typedef arg< 2, edges_of_cells_storage_type, enumtype::cells > p_orientation_of_normal;
            typedef arg< 3, cells_4d_storage_type, enumtype::cells > p_div_weights;

            auto stencil_prep = gridtools::make_computation< backend_t >(
                grid_,
                p_edge_length{} = edge_length,
                p_cell_area_reciprocal{} = cell_area_reciprocal,
                p_orientation_of_normal{} = orientation_of_normal,
                p_div_weights{} = div_weights,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_prep_functor, icosahedral_topology_t, icosahedral_topology_t::cells >(
                        p_edge_length(), p_cell_area_reciprocal(), p_orientation_of_normal(), p_div_weights())));
            stencil_prep.run();
            stencil_prep.sync_all();
        }

        {
            typedef arg< 0, edge_2d_storage_type, enumtype::edges > p_edge_length;
            typedef arg< 1, cell_2d_storage_type, enumtype::cells > p_cell_area_reciprocal;
            typedef arg< 2, edges_4d_storage_type, enumtype::edges > p_l_over_A;

            auto stencil_prep_on_edges = gridtools::make_computation< backend_t >(
                grid_,
                p_edge_length{} = edge_length,
                p_cell_area_reciprocal{} = cell_area_reciprocal,
                p_l_over_A{} = l_over_A,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_prep_functor_on_edges,
                        icosahedral_topology_t,
                        icosahedral_topology_t::edges >(p_edge_length(), p_cell_area_reciprocal(), p_l_over_A())));
            stencil_prep_on_edges.run();
            stencil_prep_on_edges.sync_all();
        }

        bool result = true;
        /*
         * stencil of div
         */

        {
            typedef arg< 0, edge_storage_type, enumtype::edges > p_in_edges;
            typedef arg< 1, cells_4d_storage_type, enumtype::cells > p_div_weights;
            typedef arg< 2, cell_storage_type, enumtype::cells > p_out_cells;

            auto stencil_ = gridtools::make_computation< backend_t >(
                grid_,
                p_in_edges{} = in_edges,
                p_div_weights{} = div_weights,
                p_out_cells{} = out_cells,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_functor, icosahedral_topology_t, icosahedral_topology_t::cells >(
                        p_in_edges(), p_div_weights(), p_out_cells())));
            stencil_.run();

            in_edges.sync();
            div_weights.sync();
            out_cells.sync();

            result = result && ver.verify(grid_, ref_cells, out_cells, halos);

#ifdef BENCHMARK
            benchmarker::run(stencil_, t_steps);
            std::cout << "div: " << stencil_.print_meter() << std::endl;
#endif
        }
        /*
         * stencil of div reduction into scalar
         */
        {
            typedef arg< 0, edge_storage_type, enumtype::edges > p_in_edges;
            typedef arg< 1, cells_4d_storage_type, enumtype::cells > p_div_weights;
            typedef arg< 2, cell_storage_type, enumtype::cells > p_out_cells;

            auto stencil_reduction_into_scalar = gridtools::make_computation< backend_t >(
                grid_,
                p_in_edges{} = in_edges,
                p_div_weights{} = div_weights,
                p_out_cells{} = out_cells,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_functor_reduction_into_scalar,
                        icosahedral_topology_t,
                        icosahedral_topology_t::cells >(p_in_edges(), p_div_weights(), p_out_cells())));
            stencil_reduction_into_scalar.run();

            in_edges.sync();
            div_weights.sync();
            out_cells.sync();

            result = result && ver.verify(grid_, ref_cells, out_cells, halos);

#ifdef BENCHMARK
            benchmarker::run(stencil_reduction_into_scalar, t_steps);
            std::cout << "reduction into scalar: " << stencil_reduction_into_scalar.print_meter() << std::endl;
#endif
        }

        /*
         * stencil of div flow convention
         */
        {
            typedef arg< 0, edge_storage_type, enumtype::edges > p_in_edges;
            typedef arg< 1, edge_2d_storage_type, enumtype::edges > p_edge_length;
            typedef arg< 2, cell_2d_storage_type, enumtype::cells > p_cell_area_reciprocal;
            typedef arg< 3, cell_storage_type, enumtype::cells > p_out_cells;

            auto stencil_flow_convention = gridtools::make_computation< backend_t >(
                grid_,
                p_in_edges{} = in_edges,
                p_edge_length{} = edge_length,
                p_cell_area_reciprocal{} = cell_area_reciprocal,
                p_out_cells{} = out_cells,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_functor_flow_convention_connectivity,
                        icosahedral_topology_t,
                        icosahedral_topology_t::cells >(
                        p_in_edges(), p_edge_length(), p_cell_area_reciprocal(), p_out_cells())));
            stencil_flow_convention.run();

            in_edges.sync();
            edge_length.sync();
            cell_area_reciprocal.sync();
            out_cells.sync();

            result = result && ver.verify(grid_, ref_cells, out_cells, halos);

#ifdef BENCHMARK
            benchmarker::run(stencil_flow_convention, t_steps);
            std::cout << "flow convention connectivity: " << stencil_flow_convention.print_meter() << std::endl;
#endif
        }

        return result;
    }
}
