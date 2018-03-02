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
 * This shows how we can use multidimensional fields with a location type.
 * We illustrate this with a storage with location type cells, that we use to
 * store a value per neighbouring edge. Therefore the storage has 5 dimension,
 * being the last one the one that stores values for each of the 3 neighbour edges
 * of a cell.
 * We use it in a user functor with a manual loop over the 3 edges.
 * We dont make use of the on_cells nor the grid topology of the icosahedral/octahedral grid here
 */
#include "backend_select.hpp"
#include "benchmarker.hpp"
#include "tools/verifier.hpp"
#include "unstructured_grid.hpp"
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

namespace soeov {

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

            // we loop over the 3 edges of a cell, and compute and store a value
            for (uint_t i = 0; i < 3; ++i) {
                eval(weight_edges(edge + i)) = eval(cell_area()) * (1 + (float_type)1.0 / (float_type)(i + 1));
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

        // instantiate a input field with location type cells
        auto cell_area = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("cell_area");
        // for the output storage we need to extend the storage with location type cells with an extra dimension
        // of length 3 (edges of a cell)
        auto weight_edges_meta = storage_info_extender()(cell_area.get_storage_info_ptr(), 3);
        using edges_of_cells_storage_type =
            backend_t::storage_traits_t::data_store_t< double, decltype(weight_edges_meta) >;
        edges_of_cells_storage_type weight_edges(weight_edges_meta, 0.0);
        edges_of_cells_storage_type ref_weights(weight_edges_meta, 0.0);

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

        auto grid_ = make_grid(di, dj, d3);

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
            // compute reference values
            unstructured_grid ugrid(d1, d2, d3);
            for (uint_t i = halo_nc; i < d1 - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < d2 - halo_mc; ++j) {
                        for (uint_t k = 0; k < d3; ++k) {
                            for (uint_t e = 0; e < 3; ++e) {
                                rv(i, c, j, k, e) = cv(i, c, j, k) * (1 + (float_type)1.0 / (float_type)(e + 1));
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
