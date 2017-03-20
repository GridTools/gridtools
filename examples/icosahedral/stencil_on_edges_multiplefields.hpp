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
 * This shows an example on how to use on_edges syntax with multiple input fields
 * (with location type edge) that are needed in the reduction over the edges of a cell
 * An typical operator that needs this functionality is the divergence where we need
 * sum_reduce(edges) {sign_edge * lengh_edge}
 * The sign of the edge indicates whether flows go inward or outward (with respect the center of the cell).
 */
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "tools/verifier.hpp"
#include "unstructured_grid.hpp"
#include "../benchmarker.hpp"

using namespace gridtools;
using namespace enumtype;

namespace soem {

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
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 1 > > in1;
        typedef in_accessor< 1, icosahedral_topology_t::edges, extent< 1 > > in2;
        typedef inout_accessor< 2, icosahedral_topology_t::edges > out;
        typedef boost::mpl::vector< in1, in2, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            auto ff = [](
                const double _in1, const double _in2, const double _res) -> double { return _in1 + _in2 * 0.1 + _res; };

            eval(out()) = eval(on_edges(ff, 0.0, in1(), in2()));
        }
    };

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        using edge_storage_type = typename icosahedral_topology_t::storage_t< icosahedral_topology_t::edges, double >;

        const uint_t halo_nc = 1;
        const uint_t halo_mc = 1;
        const uint_t halo_k = 0;
        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        auto in_edges1 = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("in1");
        auto in_edges2 = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("in2");
        auto out_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("out");
        auto ref_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("ref");

        for (int i = 0; i < d1; ++i) {
            for (int c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                for (int j = 0; j < d2; ++j) {
                    for (int k = 0; k < d3; ++k) {
                        in_edges1(i, c, j, k) = (uint_t)in_edges1.meta_data().index(
                            array< uint_t, 4 >{(uint_t)i, (uint_t)c, (uint_t)j, (uint_t)k});
                        in_edges2(i, c, j, k) = (uint_t)in_edges2.meta_data().index(
                            array< uint_t, 4 >{(uint_t)i / 2, (uint_t)c, (uint_t)j / 2, (uint_t)k / 2});
                    }
                }
            }
        }
        out_edges.initialize(0.0);
        ref_edges.initialize(0.0);

        typedef arg< 0, edge_storage_type, enumtype::edges > p_in_edges1;
        typedef arg< 1, edge_storage_type, enumtype::edges > p_in_edges2;
        typedef arg< 2, edge_storage_type, enumtype::edges > p_out_edges;

        typedef boost::mpl::vector< p_in_edges1, p_in_edges2, p_out_edges > accessor_list_t;

        gridtools::aggregator_type< accessor_list_t > domain(
            boost::fusion::make_vector(&in_edges1, &in_edges2, &out_edges));
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
                gridtools::make_stage< test_on_edges_functor, icosahedral_topology_t, icosahedral_topology_t::edges >(
                    p_in_edges1(), p_in_edges2(), p_out_edges())));
        stencil_->ready();
        stencil_->steady();
        stencil_->run();

#ifdef __CUDACC__
        out_edges.d2h_update();
        in_edges1.d2h_update();
        in_edges2.d2h_update();
#endif
        bool result = true;
        if (verify) {
            unstructured_grid ugrid(d1, d2, d3);
            for (uint_t i = halo_nc; i < d1 - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < d2 - halo_mc; ++j) {
                        for (uint_t k = 0; k < d3; ++k) {
                            auto neighbours =
                                ugrid.neighbours_of< icosahedral_topology_t::edges, icosahedral_topology_t::edges >(
                                    {i, c, j, k});
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                ref_edges(i, c, j, k) += in_edges1(*iter) + in_edges2(*iter) * 0.1;
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

            array< array< uint_t, 2 >, 4 > halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
            result = ver.verify(grid_, ref_edges, out_edges, halos);
        }
#ifdef BENCHMARK
        benchmarker::run(stencil_, t_steps);
#endif
        stencil_->finalize();

        return result;
    }
} // namespace soe
