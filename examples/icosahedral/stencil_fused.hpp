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
#include "unstructured_grid.hpp"
#include "benchmarker.hpp"
#include "backend_select.hpp"

using namespace gridtools;
using namespace enumtype;

namespace sf {

    using icosahedral_topology_t = ::gridtools::icosahedral_topology< backend_t >;

    using x_interval = axis< 1 >::full_interval;

    template < uint_t Color >
    struct test_on_edges_functor {
        typedef in_accessor< 0, icosahedral_topology_t::edges, extent< 0, 1, 0, 1 > > in;
        typedef inout_accessor< 1, icosahedral_topology_t::cells > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            auto ff = [](const double _in, const double _res) -> double { return _in + _res; };

            /**
               This interface checks that the location types are compatible with the accessors
             */
            eval(out()) = eval(on_edges(ff, 0.0, in()));
        }
    };

    template < uint_t Color >
    struct test_on_cells_functor {
        typedef in_accessor< 0, icosahedral_topology_t::cells, extent< -1, 1, -1, 1 > > in;
        typedef inout_accessor< 1, icosahedral_topology_t::cells > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            auto ff = [](const double _in, const double _res) -> double { return _in + _res; };
            /**
               This interface checks that the location types are compatible with the accessors
             */
            eval(out()) = eval(on_cells(ff, 0.0, in()));
        }
    };

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        using cell_storage_type =
            typename icosahedral_topology_t::data_store_t< icosahedral_topology_t::cells, double, halo< 2, 0, 2, 0 > >;
        using edge_storage_type =
            typename icosahedral_topology_t::data_store_t< icosahedral_topology_t::edges, double, halo< 2, 0, 2, 0 > >;

        const uint_t halo_nc = 2;
        const uint_t halo_mc = 2;
        const uint_t halo_k = 0;
        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        auto in_edges =
            icosahedral_grid.make_storage< icosahedral_topology_t::edges, double, halo< 2, 0, 2, 0 > >("in_edge");
        auto out_cells =
            icosahedral_grid.make_storage< icosahedral_topology_t::cells, double, halo< 2, 0, 2, 0 > >("out");
        auto ref_on_cells =
            icosahedral_grid.make_storage< icosahedral_topology_t::cells, double, halo< 2, 0, 2, 0 > >("ref_on_cells");
        auto ref_on_cells_tmp =
            icosahedral_grid.make_storage< icosahedral_topology_t::cells, double /* , halo<2,0,2,0> */ >(
                "ref_on_cells_tmp");

        typedef decltype(in_edges) in_edges_storage_t;
        typedef decltype(out_cells) out_cells_storage_t;
        typedef decltype(ref_on_cells_tmp) ref_on_cells_tmp_storage_t;
        typedef decltype(ref_on_cells) ref_on_cells_storage_t;

        auto iev = make_host_view(in_edges);
        for (int i = 1; i < d1 - 1; ++i) {
            for (int c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
                for (int j = 1; j < d2 - 1; ++j) {
                    for (int k = 0; k < d3; ++k) {
                        iev(i, c, j, k) = in_edges.get_storage_info_ptr()->index(i, c, j, k);
                    }
                }
            }
        }

        out_cells = out_cells_storage_t(*out_cells.get_storage_info_ptr(), 0.0);
        ref_on_cells = ref_on_cells_storage_t(*ref_on_cells.get_storage_info_ptr(), 0.0);
        ref_on_cells_tmp = ref_on_cells_tmp_storage_t(*ref_on_cells_tmp.get_storage_info_ptr(), 0.0);
        auto roctv = make_host_view(ref_on_cells_tmp);
        auto ocv = make_host_view(out_cells);
        auto rocv = make_host_view(ref_on_cells);

        typedef arg< 0, edge_storage_type, enumtype::edges > p_in_edges;
        typedef tmp_arg< 1, cell_storage_type, enumtype::cells > p_tmp_cells;
        typedef arg< 2, cell_storage_type, enumtype::cells > p_out_cells;

        halo_descriptor di{halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        halo_descriptor dj{halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        auto grid_ = make_grid(icosahedral_grid, di, dj, d3);

        auto stencil_cells = gridtools::make_computation< backend_t >(
            grid_,
            p_in_edges() = in_edges,
            p_out_cells() = out_cells,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(),
                gridtools::make_stage< test_on_edges_functor, icosahedral_topology_t, icosahedral_topology_t::cells >(
                    p_in_edges(), p_tmp_cells()),
                gridtools::make_stage< test_on_cells_functor, icosahedral_topology_t, icosahedral_topology_t::cells >(
                    p_tmp_cells(), p_out_cells())));
        stencil_cells.run();

        out_cells.sync();
        in_edges.sync();

        bool result = true;
        if (verify) {
            unstructured_grid ugrid(d1, d2, d3);
            for (uint_t i = halo_nc - 1; i < d1 - halo_nc + 1; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                    for (uint_t j = halo_mc - 1; j < d2 - halo_mc + 1; ++j) {
                        for (uint_t k = 0; k < d3; ++k) {
                            auto neighbours =
                                ugrid.neighbours_of< icosahedral_topology_t::cells, icosahedral_topology_t::edges >(
                                    {i, c, j, k});
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                roctv(i, c, j, k) += iev((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]);
                            }
                        }
                    }
                }
            }

            for (uint_t i = halo_nc; i < d1 - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < d2 - halo_mc; ++j) {
                        for (uint_t k = 0; k < d3; ++k) {
                            auto neighbours =
                                ugrid.neighbours_of< icosahedral_topology_t::cells, icosahedral_topology_t::cells >(
                                    {i, c, j, k});
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                rocv(i, c, j, k) += roctv((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]);
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
            result = ver.verify(grid_, ref_on_cells, out_cells, halos);
        }
#ifdef BENCHMARK
        benchmarker::run(stencil_cells, t_steps);
#endif
        return result;
    }

} // namespace soc
