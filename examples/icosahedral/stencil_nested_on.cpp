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

using namespace gridtools;
using namespace enumtype;

namespace nested_test {

#ifdef __CUDACC__
    using backend_t = ::gridtools::backend< Cuda, GRIDBACKEND, Block >;
#else
#ifdef BACKEND_BLOCK
    using backend_t = ::gridtools::backend< Host, GRIDBACKEND, Block >;
#else
    using backend_t = ::gridtools::backend< Host, GRIDBACKEND, Naive >;
#endif
#endif

    using icosahedral_topology_t = ::gridtools::icosahedral_topology< backend_t >;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    template < uint_t Color >
    struct nested_stencil {
        typedef in_accessor< 0, icosahedral_topology_t::cells, extent< 2 > > in_cells;
        typedef in_accessor< 1, icosahedral_topology_t::edges, extent< 1 > > in_edges;
        typedef in_accessor< 2, icosahedral_topology_t::edges, extent< 1 > > ipos;
        typedef in_accessor< 3, icosahedral_topology_t::edges, extent< 1 > > cpos;
        typedef in_accessor< 4, icosahedral_topology_t::edges, extent< 1 > > jpos;
        typedef in_accessor< 5, icosahedral_topology_t::edges, extent< 1 > > kpos;
        typedef inout_accessor< 6, icosahedral_topology_t::edges > out_edges;

        typedef boost::mpl::vector< in_cells, in_edges, ipos, cpos, jpos, kpos, out_edges > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            auto ff = [](const double _in, const double _res) -> double {
                std::cout << "INNER FF " << _in << " " << _res << " " << _in + _res + 1 << std::endl;

                return _in + _res + 1;
            };
            auto gg = [](const double _in, const double _res) -> double {
                std::cout << "MAP ON EDGES " << _in << " " << _res << " " << _in + _res + 2 << std::endl;

                return _in + _res + 2;
            };
            auto reduction = [](const double _in, const double _res) -> double {
                std::cout << "RED ON EDGES " << _in << " " << _res << " " << _in + _res + 3 << std::endl;
                return _in + _res + 3;
            };

            std::cout << "FOR i,c,j,k " << eval(ipos()) << " " << eval(jpos()) << " " << eval(cpos()) << " "
                      << eval(kpos()) << std::endl;

            //            auto x = eval(on_edges(reduction, 0.0,
            //                map(gg, in_edges(), on_cells(ff, 0.0, map(identity<double>(), in_cells())))));
            eval(on_edges(reduction, 0.0, map(gg, in_edges(), on_cells(ff, 0.0, in_cells()))));
            // auto y = eval(on_edges(reduction, 0.0, map(gg, in_edges(), on_cells(ff, 0.0, in_cells()))));
            // eval(out()) = eval(reduce_on_edges(reduction, 0.0, edges0::reduce_on_cells(gg, in()), edges1()));
        }
    };
}

using namespace nested_test;

TEST(test_stencil_nested_on, run) {

    using cell_storage_type = typename backend_t::storage_t< icosahedral_topology_t::cells, double >;
    using edge_storage_type = typename backend_t::storage_t< icosahedral_topology_t::edges, double >;

    const uint_t halo_nc = 1;
    const uint_t halo_mc = 1;
    const uint_t halo_k = 0;
    const uint_t d3 = 6 + halo_k * 2;
    const uint_t d1 = 6 + halo_nc * 2;
    const uint_t d2 = 6 + halo_mc * 2;
    icosahedral_topology_t icosahedral_grid(d1, d2, d3);

    cell_storage_type in_cells = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("in_cell");
    edge_storage_type in_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("in_edge");
    edge_storage_type out_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("out_edge");

    edge_storage_type i_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("i");
    edge_storage_type j_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("j");
    edge_storage_type c_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("c");
    edge_storage_type k_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("k");
    edge_storage_type ref_edges = icosahedral_grid.make_storage< icosahedral_topology_t::edges, double >("ref");

    in_cells.allocate();
    in_edges.allocate();
    out_edges.allocate();
    i_edges.allocate();
    j_edges.allocate();
    c_edges.allocate();
    k_edges.allocate();
    ref_edges.allocate();

    auto incv = make_host_view(in_cells);
    auto inev = make_host_view(in_edges);
    auto iv = make_host_view(i_edges);
    auto cv = make_host_view(c_edges);
    auto jv = make_host_view(j_edges);
    auto kv = make_host_view(k_edges);
    auto rv = make_host_view(ref_edges);

    for (int i = 0; i < d1; ++i) {
        for (int c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
            for (int j = 0; j < d2; ++j) {
                for (int k = 0; k < d3; ++k) {
                    if (c < icosahedral_topology_t::cells::n_colors::value) {
                        incv(i, c, j, k) = in_cells.get_storage_info_ptr()->index(i, c, j, k);
                    }
                    inev(i, c, j, k) = in_edges.get_storage_info_ptr()->index(i, c, j, k);

                    iv(i, c, j, k) = i;
                    cv(i, c, j, k) = c;
                    jv(i, c, j, k) = j;
                    kv(i, c, j, k) = k;
                    rv(i, c, j, k) = 0;
                }
            }
        }
    }

    typedef arg< 0, cell_storage_type > p_in_cells;
    typedef arg< 1, edge_storage_type > p_in_edges;
    typedef arg< 2, edge_storage_type > p_i_edges;
    typedef arg< 3, edge_storage_type > p_c_edges;
    typedef arg< 4, edge_storage_type > p_j_edges;
    typedef arg< 5, edge_storage_type > p_k_edges;
    typedef arg< 6, edge_storage_type > p_out_edges;

    typedef boost::mpl::vector< p_in_cells, p_in_edges, p_i_edges, p_c_edges, p_j_edges, p_k_edges, p_out_edges >
        accessor_list_t;

    gridtools::aggregator_type< accessor_list_t > domain(
        in_cells, in_edges, i_edges, c_edges, j_edges, k_edges, out_edges);
    array< uint_t, 5 > di = {halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
    array< uint_t, 5 > dj = {halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

    gridtools::grid< axis, icosahedral_topology_t > grid_(icosahedral_grid, di, dj);
    grid_.value_list[0] = 0;
    grid_.value_list[1] = d3 - 1;

    std::shared_ptr< gridtools::stencil > copy = gridtools::make_computation< backend_t >(
        domain,
        grid_,
        gridtools::make_multistage // mss_descriptor
        (execute< forward >(),
            gridtools::make_stage< nested_stencil, icosahedral_topology_t, icosahedral_topology_t::cells >(
                p_in_cells(), p_in_edges(), p_i_edges(), p_c_edges(), p_j_edges(), p_k_edges(), p_out_edges())));
    copy->ready();
    copy->steady();
    copy->run();
    copy->finalize();

    unstructured_grid ugrid(d1, d2, d3);
    for (uint_t i = halo_nc; i < d1 - halo_nc; ++i) {
        for (uint_t c = 0; c < icosahedral_topology_t::edges::n_colors::value; ++c) {
            for (uint_t j = halo_mc; j < d2 - halo_mc; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    std::cout << "REFFOR i,c,j,k " << iv(i, c, j, k) << " " << cv(i, c, j, k) << " " << jv(i, c, j, k)
                              << " " << kv(i, c, j, k) << std::endl;

                    double acc = 0.0;
                    auto neighbours =
                        ugrid.neighbours_of< icosahedral_topology_t::edges, icosahedral_topology_t::edges >(
                            {i, c, j, k});
                    for (auto edge_iter = neighbours.begin(); edge_iter != neighbours.end(); ++edge_iter) {
                        auto innercell_neighbours =
                            ugrid.neighbours_of< icosahedral_topology_t::edges, icosahedral_topology_t::cells >(
                                *edge_iter);
                        for (auto cell_iter = innercell_neighbours.begin(); cell_iter != innercell_neighbours.end();
                             ++cell_iter) {
                            std::cout << "REF INNER FF "
                                      << incv((*cell_iter)[0], (*cell_iter)[1], (*cell_iter)[2], (*cell_iter)[3]) << " "
                                      << acc << " "
                                      << acc +
                                             incv((*cell_iter)[0], (*cell_iter)[1], (*cell_iter)[2], (*cell_iter)[3]) +
                                             1
                                      << std::endl;

                            acc += incv((*cell_iter)[0], (*cell_iter)[1], (*cell_iter)[2], (*cell_iter)[3]) + 1;
                        }
                        std::cout << "MAP ON EDGES " << acc << " "
                                  << inev((*edge_iter)[0], (*edge_iter)[1], (*edge_iter)[2], (*edge_iter)[3]) << " "
                                  << (acc + inev((*edge_iter)[0], (*edge_iter)[1], (*edge_iter)[2], (*edge_iter)[3]) +
                                         2)
                                  << std::endl;
                    }

                    std::cout << "RED ON EDGES " << (acc + inev(i, c, j, k) + 2) << " " << rv(i, c, j, k) << " "
                              << (acc + inev(i, c, j, k) + 2) + 3 << std::endl;

                    rv(i, c, j, k) += (acc + inev(i, c, j, k) + 2) + 3;
                }
            }
        }
    }

#if FLOAT_PRECISION == 4
    verifier ver(1e-6);
#else
    verifier ver(1e-12);
#endif

    array< array< uint_t, 2 >, 4 > halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
    EXPECT_TRUE(ver.verify(grid_, ref_edges, out_edges, halos));
}
