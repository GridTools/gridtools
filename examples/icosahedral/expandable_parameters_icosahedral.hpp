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
#define FUSION_MAX_VECTOR_SIZE 40
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#include <stencil-composition/stencil-composition.hpp>
#include <tools/verifier.hpp>

namespace test_expandable_parameters_icosahedral {

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
    struct functor_exp {

        typedef vector_accessor< 0, enumtype::inout, icosahedral_topology_t::cells > parameters_out;
        typedef vector_accessor< 1, enumtype::in, icosahedral_topology_t::cells > parameters_in;
        typedef boost::mpl::vector< parameters_out, parameters_in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(parameters_out{}) = eval(parameters_in{});
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3, uint_t t) {

        using backend_t = BACKEND;
        using cell_storage_type = typename icosahedral_topology_t::storage_t< icosahedral_topology_t::cells, double >;

        icosahedral_topology_t icosahedral_grid(d1, d2, d3);

        auto storage1 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage1");
        auto storage2 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage2");
        auto storage3 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage3");
        auto storage4 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage4");
        auto storage5 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage5");

        auto storage10 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage10");
        auto storage20 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage20");
        auto storage30 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage30");
        auto storage40 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage40");
        auto storage50 = icosahedral_grid.make_storage< icosahedral_topology_t::cells, double >("storage50");

        auto sinfo = *storage1.get_storage_info_ptr();

        storage1 = cell_storage_type(sinfo, 1.);
        storage2 = cell_storage_type(sinfo, 2.);
        storage3 = cell_storage_type(sinfo, 3.);
        storage4 = cell_storage_type(sinfo, 4.);
        storage5 = cell_storage_type(sinfo, 5.);

        storage10 = cell_storage_type(sinfo, 10.);
        storage20 = cell_storage_type(sinfo, 20.);
        storage30 = cell_storage_type(sinfo, 30.);
        storage40 = cell_storage_type(sinfo, 40.);
        storage50 = cell_storage_type(sinfo, 50.);

        std::vector< decltype(storage1) > list_out_ = {storage1, storage2, storage3, storage4, storage5};

        std::vector< decltype(storage10) > list_in_ = {storage10, storage20, storage30, storage40, storage50};

        array< uint_t, 5 > di = {0, 0, 0, d1 - 1, d1};
        array< uint_t, 5 > dj = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis, icosahedral_topology_t > grid_(icosahedral_grid, di, dj);
        grid_.value_list[0] = 0;
        grid_.value_list[1] = d3 - 1;

        using p_list_out = arg< 0, std::vector< decltype(storage1) > >;
        using p_list_in = arg< 1, std::vector< decltype(storage10) > >;

        typedef boost::mpl::vector< p_list_out, p_list_in > args_t;

        aggregator_type< args_t > domain_(list_out_, list_in_);

        auto comp_ = make_computation< BACKEND >(
            expand_factor< 2 >(),
            domain_,
            grid_,
            make_multistage(enumtype::execute< enumtype::forward >(),
                make_stage< functor_exp, icosahedral_topology_t, icosahedral_topology_t::cells >(
                                p_list_out(), p_list_in())));

        comp_->ready();
        comp_->steady();
        comp_->run();
        comp_->finalize();

#if FLOAT_PRECISION == 4
        verifier ver(1e-6);
#else
        verifier ver(1e-10);
#endif

        array< array< uint_t, 2 >, 4 > halos = {{{0, 0}, {0, 0}, {0, 0}, {0, 0}}};
        bool result = ver.verify(grid_, storage1, storage10, halos);
        result = result & ver.verify(grid_, storage2, storage20, halos);
        result = result & ver.verify(grid_, storage3, storage30, halos);
        result = result & ver.verify(grid_, storage4, storage40, halos);
        result = result & ver.verify(grid_, storage5, storage50, halos);
        return result;
    }
} // namespace test_expandable_parameters_icosahedral
