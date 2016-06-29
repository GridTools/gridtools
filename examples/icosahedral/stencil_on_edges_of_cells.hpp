/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "unstructured_grid.hpp"
#include "../benchmarker.hpp"

using namespace gridtools;
using namespace enumtype;
using namespace expressions;

namespace soeov {

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
            edge_of_cell_dim::Index edge;

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

        using cell_storage_type = typename backend_t::storage_t< icosahedral_topology_t::cells, double >;

        const uint_t halo_nc = 1;
        const uint_t halo_mc = 1;
        const uint_t halo_k = 0;
        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        // instantiate a input field with location type cells
        auto cell_area = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("cell_area");
        // for the output storage we need to extend the storage with location type cells with an extra dimension
        // of length 3 (edges of a cell)
        auto weight_edges_meta = meta_storage_extender()(cell_area.meta_data(), 3);
        using edges_of_cells_storage_type =
            typename backend_t::storage_type< double, decltype(weight_edges_meta) >::type;
        edges_of_cells_storage_type weight_edges(weight_edges_meta, "edges_of_cell");
        edges_of_cells_storage_type ref_weights(weight_edges_meta, "ref_edges_of_cell");

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
        weight_edges.initialize(0.0);
        ref_weights.initialize(0.0);

        typedef arg< 0, cell_storage_type > p_cell_area;
        typedef arg< 1, edges_of_cells_storage_type > p_weight_edges;

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
            // compute reference values
            unstructured_grid ugrid(d1, d2, d3);
            for (uint_t i = halo_nc; i < d1 - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < d2 - halo_mc; ++j) {
                        for (uint_t k = 0; k < d3; ++k) {
                            for (uint_t e = 0; e < 3; ++e) {
                                ref_weights(i, c, j, k, e) =
                                    cell_area(i, c, j, k) * (1 + (float_type)1.0 / (float_type)(e + 1));
                            }
                        }
                    }
                }
            }

            verifier ver(1e-10);

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
