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
#elif defined(__AVX512F__)
#define BACKEND backend< Mic, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    using backend_t = BACKEND;
    using icosahedral_topology_t = ::gridtools::icosahedral_topology< backend_t >;

    using x_interval = axis< 1 >::full_interval;

    template < uint_t Color >
    struct test_on_edges_functor {
        typedef in_accessor< 0, icosahedral_topology_t::cells, extent< 1 > > cell_area;
        typedef inout_accessor< 1, icosahedral_topology_t::cells, 5 > weight_edges;
        typedef boost::mpl::vector< cell_area, weight_edges > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
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

        using cell_storage_type =
            typename icosahedral_topology_t::data_store_t< icosahedral_topology_t::cells, double >;

        const uint_t halo_nc = 1;
        const uint_t halo_mc = 1;
        const uint_t halo_k = 0;
        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        // area of cells is a storage with location type cells
        auto cell_area = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("cell_area");
        // we need a storage of weights with location cells, but storing 3 weights (one for each edge).
        // We extend the storage with location type cells by one more dimension with length 3
        auto weight_edges_meta = storage_info_extender()(cell_area.get_storage_info_ptr(), 3);
        using edges_of_cells_storage_type =
            backend_t::storage_traits_t::data_store_t< double, decltype(weight_edges_meta) >;
        // allocate the weight on edges of cells and the reference values
        edges_of_cells_storage_type weight_edges(weight_edges_meta, 0.0);
        edges_of_cells_storage_type ref_weights(weight_edges_meta, 0.0);

        // dummy initialization of input values of the cell areas
        auto cv = make_host_view(cell_area);
        for (int i = 0; i < d1; ++i) {
            for (int c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                for (int j = 0; j < d2; ++j) {
                    for (int k = 0; k < d3; ++k) {
                        cv(i, c, j, k) = (uint_t)cell_area.get_storage_info_ptr()->index(i, c, j, k);
                    }
                }
            }
        }

        typedef arg< 0, cell_storage_type, enumtype::cells > p_cell_area;
        typedef arg< 1, edges_of_cells_storage_type, enumtype::cells > p_weight_edges;

        typedef boost::mpl::vector< p_cell_area, p_weight_edges > accessor_list_t;

        gridtools::aggregator_type< accessor_list_t > domain(cell_area, weight_edges);

        halo_descriptor di{halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        halo_descriptor dj{halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        auto grid_ = make_grid(icosahedral_grid, di, dj, d3);

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

        cell_area.sync();
        weight_edges.sync();

        bool result = true;
        if (verify) {
            auto rv = make_host_view(ref_weights);
            auto wv = make_host_view(weight_edges);
            // compute the reference values of the weights on edges of cells
            unstructured_grid ugrid(d1, d2, d3);
            for (uint_t i = halo_nc; i < d1 - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < d2 - halo_mc; ++j) {
                        for (uint_t k = 0; k < d3; ++k) {

                            auto neighbours =
                                ugrid.neighbours_of< icosahedral_topology_t::cells, icosahedral_topology_t::cells >(
                                    {i, c, j, k});
                            ushort_t e = 0;
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                rv(i, c, j, k, e) = cv((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]) / cv(i, c, j, k);
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
