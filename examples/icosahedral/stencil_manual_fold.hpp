/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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

/*
 * This example demonstrates how to retrieve the connectivity information of the
 * icosahedral/octahedral grid in the user functor. This is useful for example when
 * we need to operate on fields with a double location, for which the on_cells, on_edges
 * syntax has limitations, as it requires make use of the eval object, which is not
 * resolved in the lambdas passed to the on_cells syntax.
 * The example shown here computes a value for each edge of a cells. Therefore the primary
 * location type of the output field is cells, however we do not store a scalar value, but
 * a value per edge of each cell (i.e. 3 values).
 * The storage is therefore a 5 dimensional field with indices (i, c, j, k, edge_number)
 * where the last has the range [0,2]
 *
 */

#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "unstructured_grid.hpp"
#include "../benchmarker.hpp"

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

namespace smf {

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    using backend_t = BACKEND;
    using icosahedral_topology_t = ::gridtools::icosahedral_topology< backend_t >;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    template < uint_t Color >
    struct test_on_edges_functor {
        typedef in_accessor< 0, icosahedral_topology_t::cells, extent< 1 > > cell_area;
        typedef inout_accessor< 1, icosahedral_topology_t::cells, 5 > weight_edges;
        typedef boost::mpl::vector< cell_area, weight_edges > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            using edge_of_cell_dim = dimension< 5 >;
            edge_of_cell_dim edge;

            // retrieve the array of neighbor offsets. This is an array with length 3 (number of neighbors).
            constexpr auto neighbors_offsets = connectivity< cells, cells, Color >::offsets();
            ushort_t e = 0;
            // loop over all neighbours. Each iterator (neighbor_offset) is a position offset, i.e. an array with length
            // 4
            for (auto neighbor_offset : neighbors_offsets) {
                eval(weight_edges(edge + e)) = eval(cell_area(neighbor_offset)) / eval(cell_area());
                e++;
            }
        }
    };

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        using cell_storage_type = typename icosahedral_topology_t::storage_t< icosahedral_topology_t::cells, double >;

        const uint_t halo_nc = 1;
        const uint_t halo_mc = 1;
        const uint_t halo_k = 0;
        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        // area of cells is a storage with location type cells
        auto cell_area = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("cell_area");
        // we need a storage of weights with location cells, but storing 3 weights (one for each edge).
        // We extend the storage with location type cells by one more dimension with length 3
        auto weight_edges_meta = meta_storage_extender()(cell_area.meta_data(), 3);
        using edges_of_cells_storage_type =
            typename backend_t::storage_type< double, decltype(weight_edges_meta) >::type;
        // allocate the weight on edges of cells and the reference values
        edges_of_cells_storage_type weight_edges(weight_edges_meta, "edges_of_cell");
        edges_of_cells_storage_type ref_weights(weight_edges_meta, "ref_edges_of_cell");

        // dummy initialization of input values of the cell areas
        for (int i = 0; i < d1; ++i) {
            for (int c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                for (int j = 0; j < d2; ++j) {
                    for (int k = 0; k < d3; ++k) {
                        cell_area(i, c, j, k) = (uint_t)cell_area.meta_data().index(
                            array< uint_t, 4 >{(uint_t)i, (uint_t)c, (uint_t)j, (uint_t)k});
                    }
                }
            }
        }
        // default initialize output and reference.
        weight_edges.initialize(0.0);
        ref_weights.initialize(0.0);

        typedef arg< 0, cell_storage_type, enumtype::cells > p_cell_area;
        typedef arg< 1, edges_of_cells_storage_type, enumtype::edges > p_weight_edges;

        typedef boost::mpl::vector< p_cell_area, p_weight_edges > accessor_list_t;

        gridtools::aggregator_type< accessor_list_t > domain(boost::fusion::make_vector(&cell_area, &weight_edges));
        array< uint_t, 5 > di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        array< uint_t, 5 > dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        gridtools::grid< axis, icosahedral_topology_t > grid_(icosahedral_grid, di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        auto stencil_ = gridtools::make_computation< backend_t >(
            domain,
            grid_,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(),
                gridtools::make_stage< test_on_edges_functor, icosahedral_topology_t, icosahedral_topology_t::cells >(
                    p_cell_area(), p_weight_edges())));
        stencil_->ready();
        stencil_->steady();
        stencil_->run();

#ifdef __CUDACC__
        cell_area.d2h_update();
        weight_edges.d2h_update();
#endif

        bool result = true;
        if (verify) {
            // compute the reference values of the weights on edges of cells
            unstructured_grid ugrid(d1, d2, d3);
            for (uint_t i = halo_nc; i < d1 - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < d2 - halo_mc; ++j) {
                        for (uint_t k = 0; k < d3; ++k) {

                            auto neighbours =
                                ugrid.neighbours_of< icosahedral_topology_t::cells, icosahedral_topology_t::cells >(
                                    {i, c, j, k});
                            ushort_t e = 0;
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                ref_weights(i, c, j, k, e) = cell_area(*iter) / cell_area(i, c, j, k);
                                ++e;
                            }
                        }
                    }
                }
            }

#if FLOAT_PRECISION == 4
            verifier ver(1e-6);
#else
            verifier ver(1e-10);
#endif

            array< array< uint_t, 2 >, 5 > halos = {
                {{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}, {0, 0}}};
            result = ver.verify(grid_, ref_weights, weight_edges, halos);
        }

#ifdef BENCHMARK
        benchmarker::run(stencil_, t_steps);
#endif
        stencil_->finalize();
        return result;
    }
} // namespace soeov
