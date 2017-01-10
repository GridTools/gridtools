/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

namespace test_copy_stencil_icosahedral {

    using namespace gridtools;
    using namespace expressions;
    using namespace enumtype;

    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

#ifdef __CUDACC__
#define BACKEND backend< enumtype::Cuda, GRIDBACKEND, enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< enumtype::Host, GRIDBACKEND, enumtype::Block >
#else
#define BACKEND backend< enumtype::Host, GRIDBACKEND, enumtype::Naive >
#endif
#endif

    using icosahedral_topology_t = ::gridtools::icosahedral_topology< BACKEND >;

    template < uint_t Color >
    struct functor_copy {

        typedef accessor< 0, enumtype::inout, icosahedral_topology_t::cells > out;
        typedef accessor< 1, enumtype::in, icosahedral_topology_t::cells > in;
        typedef boost::mpl::vector< out, in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out{}) = eval(in{});
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t) {

        using backend_t = BACKEND;
        using cell_storage_type = typename icosahedral_topology_t::storage_t< icosahedral_topology_t::cells, double >;

        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        auto in_cells = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("in");
        auto out_cells = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("out");
        auto ref_cells = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("ref");

        for (int i = 1; i < d1 - 1; ++i) {
            for (int c = 0; c < icosahedral_topology_t::cells::n_colors::value; ++c) {
                for (int j = 1; j < d2 - 1; ++j) {
                    for (int k = 0; k < d3; ++k) {
                        in_cells(i, c, j, k) =
                            in_cells.meta_data().index(array< uint_t, 4 >{(uint_t)i, (uint_t)c, (uint_t)j, (uint_t)k});
                    }
                }
            }
        }
        out_cells.initialize(0.0);
        ref_cells.initialize(0.0);
        auto storage1 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage1");

        auto storage10 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage10");

        storage1.initialize(1.);

        storage10.initialize(10.);

        array< uint_t, 5 > di = {0, 0, 0, d1 - 1, d1};
        array< uint_t, 5 > dj = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis, icosahedral_topology_t > grid_(icosahedral_grid, di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        using p_out = arg< 0, decltype(storage1) >;
        using p_in = arg< 1, decltype(storage10) >;

        typedef boost::mpl::vector< p_out, p_in > args_t;

        aggregator_type< args_t > domain_((p_out() = storage1), (p_in() = storage10));

        auto comp_ = make_computation< BACKEND >(
            domain_,
            grid_,
            make_multistage(enumtype::execute< enumtype::forward >(),
                make_stage< functor_copy, icosahedral_topology_t, icosahedral_topology_t::cells >(p_out(), p_in())));

        comp_->ready();
        comp_->steady();
        comp_->run();
        comp_->finalize();

        verifier ver(1e-10);

        array< array< uint_t, 2 >, 4 > halos = {{{0, 0}, {0, 0}, {0, 0}, {0, 0}}};
        bool result = ver.verify(grid_, storage1, storage10, halos);

        return result;
    }
} // namespace test_expandable_parameters_icosahedral
