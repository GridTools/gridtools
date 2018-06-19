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
#include "backend_select.hpp"
#include "benchmarker.hpp"
#include "unstructured_grid.hpp"
#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/verifier.hpp>

using namespace gridtools;
using namespace enumtype;

namespace sov {

    using icosahedral_topology_t = ::gridtools::icosahedral_topology<backend_t>;

    using x_interval = axis<1>::full_interval;

    template <uint_t Color>
    struct test_on_vertices_functor {
        typedef in_accessor<0, icosahedral_topology_t::vertices, extent<-1, 1, -1, 1>> in;
        typedef inout_accessor<1, icosahedral_topology_t::vertices> out;
        typedef boost::mpl::vector2<in, out> arg_list;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            auto ff = [](const double _in, const double _res) -> double { return _in + _res; };

            /**
               This interface checks that the location types are compatible with the accessors
             */
            eval(out()) = eval(on_vertices(ff, 0.0, in()));
        }
    };

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        constexpr uint_t halo_nc = 1;
        constexpr uint_t halo_mc = 1;

        using vertex_storage_type = typename icosahedral_topology_t::
            data_store_t<icosahedral_topology_t::vertices, double, halo<halo_nc, 0, halo_mc, 0>>;

        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        auto in_vertices =
            icosahedral_grid.make_storage<icosahedral_topology_t::vertices, double, halo<halo_nc, 0, halo_mc, 0>>("in");
        auto out_vertices =
            icosahedral_grid.make_storage<icosahedral_topology_t::vertices, double, halo<halo_nc, 0, halo_mc, 0>>(
                "out");
        auto ref_vertices =
            icosahedral_grid.make_storage<icosahedral_topology_t::vertices, double, halo<halo_nc, 0, halo_mc, 0>>(
                "ref");
        auto inv = make_host_view(in_vertices);
        auto outv = make_host_view(out_vertices);
        auto refv = make_host_view(ref_vertices);

        for (int i = 0; i < d1; ++i) {
            for (int c = 0; c < icosahedral_topology_t::vertices::n_colors::value; ++c) {
                for (int j = 0; j < d2; ++j) {
                    for (int k = 0; k < d3; ++k) {
                        inv(i, c, j, k) = (uint_t)in_vertices.get_storage_info_ptr()->index(i, c, j, k);
                        outv(i, c, j, k) = 0.0;
                        refv(i, c, j, k) = 0.0;
                    }
                }
            }
        }

        typedef arg<0, vertex_storage_type, enumtype::vertices> p_in_vertices;
        typedef arg<1, vertex_storage_type, enumtype::vertices> p_out_vertices;

        halo_descriptor di{halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
        halo_descriptor dj{halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

        auto grid_ = make_grid(icosahedral_grid, di, dj, d3);

        auto stencil_ = gridtools::make_computation<backend_t>(grid_,
            p_in_vertices() = in_vertices,
            p_out_vertices() = out_vertices,
            gridtools::make_multistage // mss_descriptor
            (execute<forward>(),
                gridtools::make_stage<test_on_vertices_functor,
                    icosahedral_topology_t,
                    icosahedral_topology_t::vertices>(p_in_vertices(), p_out_vertices())));
        stencil_.run();

        out_vertices.sync();
        in_vertices.sync();

        bool result = true;
        if (verify) {
            unstructured_grid ugrid(d1, d2, d3);
            for (uint_t i = halo_nc; i < d1 - halo_nc; ++i) {
                for (uint_t c = 0; c < icosahedral_topology_t::vertices::n_colors::value; ++c) {
                    for (uint_t j = halo_mc; j < d2 - halo_mc; ++j) {
                        for (uint_t k = 0; k < d3; ++k) {
                            auto neighbours =
                                ugrid.neighbours_of<icosahedral_topology_t::vertices, icosahedral_topology_t::vertices>(
                                    {i, c, j, k});
                            for (auto iter = neighbours.begin(); iter != neighbours.end(); ++iter) {
                                refv(i, c, j, k) += inv((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]);
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

            array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {0, 0}}};
            result = ver.verify(grid_, ref_vertices, out_vertices, halos);
        }
#ifdef BENCHMARK
        benchmarker::run(stencil_, t_steps);
#endif
        return result;
    }

} // namespace sov
