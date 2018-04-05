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
#pragma once
#include <stencil-composition/stencil-composition.hpp>
#include <tools/verifier.hpp>
#include "backend_select.hpp"

namespace test_copy_stencil_icosahedral {

    using namespace gridtools;
    using namespace expressions;
    using namespace enumtype;

    using icosahedral_topology_t = icosahedral_topology< backend_t >;
    using x_interval = axis< 1 >::full_interval;

    template < uint_t Color >
    struct functor_copy {

        typedef accessor< 0, enumtype::inout, icosahedral_topology_t::cells > out;
        typedef accessor< 1, enumtype::in, icosahedral_topology_t::cells > in;
        typedef boost::mpl::vector< out, in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out{}) = eval(in{});
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t) {

        using cell_storage_type =
            typename icosahedral_topology_t::data_store_t< icosahedral_topology_t::cells, double >;

        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        auto storage1 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage1");
        auto storage10 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage10");

        auto icv = make_host_view(storage1);
        auto ocv = make_host_view(storage10);
        for (int i = 0; i < d1; ++i) {
            for (int c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                for (int j = 0; j < d2; ++j) {
                    for (int k = 0; k < d3; ++k) {
                        icv(i, c, j, k) = storage1.get_storage_info_ptr()->index(i, c, j, k);
                        ocv(i, c, j, k) = 10.;
                    }
                }
            }
        }

        auto grid_ = make_grid(icosahedral_grid, d1, d2, d3);

        using p_out = arg< 0, decltype(storage1), enumtype::cells >;
        using p_in = arg< 1, decltype(storage10), enumtype::cells >;

        auto comp_ = make_computation< backend_t >(
            grid_,
            p_out() = storage1,
            p_in() = storage10,
            make_multistage(enumtype::execute< enumtype::forward >(),
                make_stage< functor_copy, icosahedral_topology_t, icosahedral_topology_t::cells >(p_out(), p_in())));

        comp_.run();
        comp_.sync_all();

#if FLOAT_PRECISION == 4
        verifier ver(1e-6);
#else
        verifier ver(1e-10);
#endif

        array< array< uint_t, 2 >, 4 > halos = {{{0, 0}, {0, 0}, {0, 0}, {0, 0}}};
        bool result = ver.verify(grid_, storage1, storage10, halos);

        return result;
    }
} // namespace test_expandable_parameters_icosahedral
