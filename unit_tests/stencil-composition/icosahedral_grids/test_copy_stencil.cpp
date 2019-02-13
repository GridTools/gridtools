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
#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/verifier.hpp>

using namespace gridtools;

namespace cs_test {

    using icosahedral_topology_t = ::gridtools::icosahedral_topology<backend_t>;

    using x_interval = axis<1>::full_interval;

    template <uint_t Color>
    struct test_functor {
        typedef in_accessor<0, icosahedral_topology_t::cells, extent<1>> in;
        typedef inout_accessor<1, icosahedral_topology_t::cells> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };
} // namespace cs_test

using namespace cs_test;

TEST(test_copy_stencil, run) {

    using cell_storage_type = typename icosahedral_topology_t::data_store_t<icosahedral_topology_t::cells, double>;

    const uint_t halo_nc = 1;
    const uint_t halo_mc = 2;
    const uint_t halo_k = 0;
    const uint_t d3 = 6 + halo_k * 2;
    const uint_t d1 = 6 + halo_nc * 2;
    const uint_t d2 = 12 + halo_mc * 2;
    icosahedral_topology_t icosahedral_grid(d1, d2, d3);

    cell_storage_type in_cells = icosahedral_grid.make_storage<icosahedral_topology_t::cells, double>("in");
    cell_storage_type out_cells = icosahedral_grid.make_storage<icosahedral_topology_t::cells, double>("out");

    auto inv = make_host_view(in_cells);
    auto outv = make_host_view(out_cells);

    for (int i = 0; i < d1; ++i) {
        for (int c = 0; c < 2; ++c) {
            for (int j = 0; j < d2; ++j) {
                for (int k = 0; k < d3; ++k) {
                    inv(i, c, j, k) = i + c * 100 + j * 10000 + k * 1000000;
                    outv(i, c, j, k) = 1.1;
                }
            }
        }
    }

    typedef arg<0, cell_storage_type, enumtype::cells> p_in_cells;
    typedef arg<1, cell_storage_type, enumtype::cells> p_out_cells;

    halo_descriptor di{halo_nc, halo_nc, halo_nc, d1 - halo_nc - 1, d1};
    halo_descriptor dj{halo_mc, halo_mc, halo_mc, d2 - halo_mc - 1, d2};

    gridtools::grid<axis<1>::axis_interval_t, icosahedral_topology_t> grid_(
        icosahedral_grid, di, dj, {halo_k, d3 - halo_k});

    auto copy = gridtools::make_computation<backend_t>(grid_,
        p_in_cells() = in_cells,
        p_out_cells() = out_cells,
        gridtools::make_multistage // mss_descriptor
        (execute::forward(),
            gridtools::make_stage<test_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(
                p_in_cells(), p_out_cells())));
    copy.run();

    in_cells.sync();
    out_cells.sync();

    verifier ver(1e-10);
    array<array<uint_t, 2>, 4> halos = {{{halo_nc, halo_nc}, {0, 0}, {halo_mc, halo_mc}, {halo_k, halo_k}}};
    EXPECT_TRUE(ver.verify(grid_, in_cells, out_cells, halos));
}
