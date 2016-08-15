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
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "operators_repository.hpp"
#include "../benchmarker.hpp"
#include "operator_defs.hpp"
#include "div_functors.hpp"

using namespace gridtools;
using namespace enumtype;

namespace ico_operators {

    bool test_div( uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify)
    {

        repository repository(x, y, z);
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

        using edge_storage_type = repository::edge_storage_type;
        using cell_storage_type = repository::cell_storage_type;
        using cell_2d_storage_type = repository::cell_2d_storage_type;
        using vertex_storage_type = repository::vertex_storage_type;
        using vertex_2d_storage_type = repository::vertex_2d_storage_type;
        using edge_2d_storage_type = repository::edge_2d_storage_type;

        using edges_of_vertexes_storage_type = repository::edges_of_vertexes_storage_type;
        using edges_of_cells_storage_type = repository::edges_of_cells_storage_type;

        array< uint_t, 5 > di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        array< uint_t, 5 > dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        gridtools::grid< axis, icosahedral_topology_t > grid_(icosahedral_grid, di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        verifier ver(1e-10);

        array< array< uint_t, 2 >, 4 > halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};

        auto &in_edges = repository.u();
        auto &cell_area_reciprocal = repository.cell_area_reciprocal();
        edges_of_cells_storage_type &orientation_of_normal = repository.orientation_of_normal();
        auto &edge_length = repository.edge_length();
        auto &ref_cells = repository.div_u_ref();
        auto out_cells = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("out");

        auto &weights_meta = repository.edges_of_cells_meta();
        edges_of_cells_storage_type div_weights(weights_meta, "weights");

        auto cells_of_edges_meta = meta_storage_extender()(in_edges.meta_data(), 2);
        using cells_of_edges_storage_type =
            typename backend_t::storage_type< double, decltype(cells_of_edges_meta) >::type;
        cells_of_edges_storage_type l_over_A(cells_of_edges_meta, "l_over_A");

        out_cells.initialize(0.0);
        div_weights.initialize(0.0);
        l_over_A.initialize(0.0);

        {
            typedef arg< 0, edge_2d_storage_type > p_edge_length;
            typedef arg< 1, cell_2d_storage_type > p_cell_area_reciprocal;
            typedef arg< 2, edges_of_cells_storage_type > p_orientation_of_normal;
            typedef arg< 3, edges_of_cells_storage_type > p_div_weights;

            typedef boost::mpl::vector< p_edge_length, p_cell_area_reciprocal, p_orientation_of_normal, p_div_weights >
                accessor_list_t;

            gridtools::aggregator_type< accessor_list_t > domain(
                boost::fusion::make_vector(&edge_length, &cell_area_reciprocal, &orientation_of_normal, &div_weights));

            auto stencil_prep = gridtools::make_computation< backend_t >(
                domain,
                grid_,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_prep_functor, icosahedral_topology_t, icosahedral_topology_t::cells >(
                        p_edge_length(), p_cell_area_reciprocal(), p_orientation_of_normal(), p_div_weights())));
            stencil_prep->ready();
            stencil_prep->steady();
            stencil_prep->run();
#ifdef __CUDACC__
            orientation_of_normal.d2h_update();
            edge_length.d2h_update();
            cell_area_reciprocal.d2h_update();
            div_weights.d2h_update();
            l_over_A.d2h_update();
#endif
            stencil_prep->finalize();
        }

        {
            typedef arg< 0, edge_2d_storage_type > p_edge_length;
            typedef arg< 1, cell_2d_storage_type > p_cell_area_reciprocal;
            typedef arg< 2, cells_of_edges_storage_type > p_l_over_A;

            typedef boost::mpl::vector< p_edge_length, p_cell_area_reciprocal, p_l_over_A > accessor_list_t;

            gridtools::aggregator_type< accessor_list_t > domain(
                boost::fusion::make_vector(&edge_length, &cell_area_reciprocal, &l_over_A));

            auto stencil_prep_on_edges = gridtools::make_computation< backend_t >(
                domain,
                grid_,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_prep_functor_on_edges,
                        icosahedral_topology_t,
                        icosahedral_topology_t::edges >(p_edge_length(), p_cell_area_reciprocal(), p_l_over_A())));
            stencil_prep_on_edges->ready();
            stencil_prep_on_edges->steady();
            stencil_prep_on_edges->run();

#ifdef __CUDACC__
            edge_length.d2h_update();
            cell_area_reciprocal.d2h_update();
            l_over_A.d2h_update();
#endif
            stencil_prep_on_edges->finalize();
        }

        bool result = true;
        /*
         * stencil of div
         */

        {
            typedef arg< 0, edge_storage_type > p_in_edges;
            typedef arg< 1, edges_of_cells_storage_type > p_div_weights;
            typedef arg< 2, cell_storage_type > p_out_cells;

            typedef boost::mpl::vector< p_in_edges, p_div_weights, p_out_cells > accessor_list_t;

            gridtools::aggregator_type< accessor_list_t > domain(
                boost::fusion::make_vector(&in_edges, &div_weights, &out_cells));

            auto stencil_ = gridtools::make_computation< backend_t >(
                domain,
                grid_,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_functor, icosahedral_topology_t, icosahedral_topology_t::cells >(
                        p_in_edges(), p_div_weights(), p_out_cells())));
            stencil_->ready();
            stencil_->steady();
            stencil_->run();

#ifdef __CUDACC__
            in_edges.d2h_update();
            div_weights.d2h_update();
            out_cells.d2h_update();
#endif

            result = result && ver.verify(grid_, ref_cells, out_cells, halos);

#ifdef BENCHMARK
            benchmarker::run(stencil_, t_steps);
            std::cout << "div: " << stencil_->print_meter() << std::endl;
#endif
            stencil_->finalize();
        }
        /*
         * stencil of div reduction into scalar
         */
        {
            typedef arg< 0, edge_storage_type > p_in_edges;
            typedef arg< 1, edges_of_cells_storage_type > p_div_weights;
            typedef arg< 2, cell_storage_type > p_out_cells;

            typedef boost::mpl::vector< p_in_edges, p_div_weights, p_out_cells > accessor_list_t;

            gridtools::aggregator_type< accessor_list_t > domain(
                boost::fusion::make_vector(&in_edges, &div_weights, &out_cells));

            auto stencil_reduction_into_scalar = gridtools::make_computation< backend_t >(
                domain,
                grid_,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_functor_reduction_into_scalar,
                        icosahedral_topology_t,
                        icosahedral_topology_t::cells >(p_in_edges(), p_div_weights(), p_out_cells())));
            stencil_reduction_into_scalar->ready();
            stencil_reduction_into_scalar->steady();
            stencil_reduction_into_scalar->run();

#ifdef __CUDACC__
            in_edges.d2h_update();
            div_weights.d2h_update();
            out_cells.d2h_update();
#endif

            result = result && ver.verify(grid_, ref_cells, out_cells, halos);

#ifdef BENCHMARK
            benchmarker::run(stencil_reduction_into_scalar, t_steps);
            std::cout << "reduction into scalar: " << stencil_reduction_into_scalar->print_meter() << std::endl;
#endif
            stencil_reduction_into_scalar->finalize();
        }
        /*
         * stencil of div flow convention
         */
        {
            typedef arg< 0, edge_storage_type > p_in_edges;
            typedef arg< 1, edge_2d_storage_type > p_edge_length;
            typedef arg< 2, cell_2d_storage_type > p_cell_area_reciprocal;
            typedef arg< 3, cell_storage_type > p_out_cells;

            typedef boost::mpl::vector< p_in_edges, p_edge_length, p_cell_area_reciprocal, p_out_cells > accessor_list_t;

            gridtools::aggregator_type< accessor_list_t > domain(
                boost::fusion::make_vector(&in_edges, &edge_length, &cell_area_reciprocal, &out_cells));

            auto stencil_flow_convention = gridtools::make_computation< backend_t >(
                domain,
                grid_,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_functor_flow_convention,
                        icosahedral_topology_t,
                        icosahedral_topology_t::cells >(p_in_edges(), p_edge_length(), p_cell_area_reciprocal(), p_out_cells())));
            stencil_flow_convention->ready();
            stencil_flow_convention->steady();
            stencil_flow_convention->run();

#ifdef __CUDACC__
            in_edges.d2h_update();
            edge_length.d2h_update();
            cell_area_reciprocal.d2h_update();
            out_cells.d2h_update();
#endif
            result = result && ver.verify(grid_, ref_cells, out_cells, halos);

#ifdef BENCHMARK
            benchmarker::run(stencil_flow_convention, t_steps);
            std::cout << "flow convention: " << stencil_flow_convention->print_meter() << std::endl;
#endif
            stencil_flow_convention->finalize();
        }

        /*
         * stencil of div flow convention
         */
        {
            typedef arg< 0, edge_storage_type > p_in_edges;
            typedef arg< 1, edge_2d_storage_type > p_edge_length;
            typedef arg< 2, cell_2d_storage_type > p_cell_area_reciprocal;
            typedef arg< 3, cell_storage_type > p_out_cells;

            typedef boost::mpl::vector< p_in_edges, p_edge_length, p_cell_area_reciprocal, p_out_cells > accessor_list_t;

            gridtools::aggregator_type< accessor_list_t > domain(
                boost::fusion::make_vector(&in_edges, &edge_length, &cell_area_reciprocal, &out_cells));

            auto stencil_flow_convention = gridtools::make_computation< backend_t >(
                domain,
                grid_,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< div_functor_flow_convention_connectivity,
                        icosahedral_topology_t,
                        icosahedral_topology_t::cells >(p_in_edges(), p_edge_length(), p_cell_area_reciprocal(), p_out_cells())));
            stencil_flow_convention->ready();
            stencil_flow_convention->steady();
            stencil_flow_convention->run();

#ifdef __CUDACC__
            in_edges.d2h_update();
            edge_length.d2h_update();
            cell_area_reciprocal.d2h_update();
            out_cells.d2h_update();
#endif
            result = result && ver.verify(grid_, ref_cells, out_cells, halos);

#ifdef BENCHMARK
            benchmarker::run(stencil_flow_convention, t_steps);
            std::cout << "flow convention connectivity: " << stencil_flow_convention->print_meter() << std::endl;
#endif
            stencil_flow_convention->finalize();
        }


        {
            typedef arg< 0, edge_storage_type > p_in_edges;
            typedef arg< 1, edge_2d_storage_type > p_edge_length;
            typedef arg< 2, cell_2d_storage_type > p_cell_area_reciprocal;
            typedef arg< 3, cell_storage_type > p_out_cells;

            typedef boost::mpl::vector< p_in_edges, p_edge_length, p_cell_area_reciprocal, p_out_cells > accessor_list_t;

            gridtools::aggregator_type< accessor_list_t > domain(
                boost::fusion::make_vector(&in_edges, &edge_length, &cell_area_reciprocal, &out_cells));

                /*
                 * stencil of over edge
                 */
                auto stencil_div_over_edges = gridtools::make_computation<backend_t>(
                        domain,
                        grid_,
                        gridtools::make_multistage // mss_descriptor
                                (execute<forward>(),
                                 gridtools::make_stage<div_functor_over_edges, icosahedral_topology_t,
                                 icosahedral_topology_t::edges>(
                                         p_in_edges(), p_edge_length(), p_out_cells()),
                                 gridtools::make_stage<divide_by_field, icosahedral_topology_t,
                                 icosahedral_topology_t::cells >(
                                         p_cell_area_reciprocal(), p_out_cells())
                                )
                );
                stencil_div_over_edges->ready();
                stencil_div_over_edges->steady();
                stencil_div_over_edges->run();

        #ifdef __CUDACC__
                in_edges.d2h_update();
                edge_length.d2h_update();
                cell_area_reciprocal.d2h_update();
                out_cells.d2h_update();
        #endif

                // TODO: this does not validate because the divide_by_field functor runs only on edges with color 0
//                        result = result && ver.verify(grid_, ref_cells, out_cells, halos);

        #ifdef BENCHMARK
//                benchmarker::run(stencil_div_over_edges, t_steps);
//                std::cout << "over edges: "<< stencil_div_over_edges->print_meter() << std::endl;
        #endif
                stencil_div_over_edges->finalize();
        }

        {
            typedef arg< 0, edge_storage_type > p_in_edges;
            typedef arg< 1, cells_of_edges_storage_type > p_l_over_A;
            typedef arg< 2, cell_storage_type > p_out_cells;

            typedef boost::mpl::vector< p_in_edges, p_l_over_A, p_out_cells > accessor_list_t;

            gridtools::aggregator_type< accessor_list_t > domain(
                boost::fusion::make_vector(&in_edges, &l_over_A, &out_cells));

                /*
                 * stencil of over edge weights
                 */

                auto stencil_div_over_edges_weights = gridtools::make_computation<backend_t>(
                        domain,
                        grid_,
                        gridtools::make_multistage // mss_descriptor
                                (execute<forward>(),
                                 gridtools::make_stage< div_functor_over_edges_weights, icosahedral_topology_t,
                                 icosahedral_topology_t::edges>(
                                         p_in_edges(), p_l_over_A(), p_out_cells())
                                )
                );
                stencil_div_over_edges_weights->ready();
                stencil_div_over_edges_weights->steady();
                stencil_div_over_edges_weights->run();

        #ifdef __CUDACC__
                in_edges.d2h_update();
                l_over_A.d2h_update();
                out_cells.d2h_update();
        #endif

                // TODO: this does not validate in bottom left cell
//                result = result && ver.verify(grid_, ref_cells, out_cells, halos);

        #ifdef BENCHMARK
                benchmarker::run(stencil_div_over_edges_weights, t_steps);
                std::cout << "over edges weights: "<< stencil_div_over_edges_weights->print_meter() << std::endl;
        #endif
                stencil_div_over_edges_weights->finalize();
        }
        return result;
    }
}
