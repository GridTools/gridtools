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
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/stencil-functions/stencil-functions.hpp>
#include <tools/verifier.hpp>

using namespace gridtools;
using namespace gridtools::enumtype;
using namespace gridtools::expressions;

struct copy_functor {
    typedef vector_accessor< 0, enumtype::inout > out;
    typedef vector_accessor< 1, enumtype::in > in;

    typedef boost::mpl::vector< out, in > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out{}) = eval(in{});
    }
};

struct call_proc_copy_functor {
    typedef vector_accessor< 0, enumtype::inout > out;
    typedef vector_accessor< 1, enumtype::in > in;

    typedef boost::mpl::vector< out, in > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        call_proc< copy_functor >::with(eval, out(), in());
    }
};

struct call_copy_functor {
    typedef vector_accessor< 0, enumtype::inout > out;
    typedef vector_accessor< 1, enumtype::in > in;

    typedef boost::mpl::vector< out, in > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval) {
        eval(out()) = call< copy_functor >::with(eval, in());
    }
};

class expandable_parameters : public testing::Test {
  protected:
#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif
    using axis = interval< level< 0, -1 >, level< 1, 1 > >;

    const uint_t d1 = 13;
    const uint_t d2 = 9;
    const uint_t d3 = 7;
    const uint_t halo_size = 0;

    typedef gridtools::storage_traits< BACKEND::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
    typedef gridtools::storage_traits< BACKEND::s_backend_id >::data_store_t< float_type, storage_info_t > data_store_t;

    storage_info_t meta_;

    halo_descriptor di;
    halo_descriptor dj;
    gridtools::grid< axis > grid;

    verifier verifier_;
    array< array< uint_t, 2 >, 3 > verifier_halos;

    data_store_t in_1;
    data_store_t in_2;
    data_store_t in_3;
    data_store_t in_4;
    data_store_t in_5;

    data_store_t out_1;
    data_store_t out_2;
    data_store_t out_3;
    data_store_t out_4;
    data_store_t out_5;

    std::vector< data_store_t > in;
    std::vector< data_store_t > out;

    typedef arg< 0, std::vector< data_store_t > > p_in;
    typedef arg< 1, std::vector< data_store_t > > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    aggregator_type< accessor_list > domain;

    expandable_parameters()
        : meta_(d1, d2, d3), di(halo_size, halo_size, halo_size, d1 - halo_size - 1, d1),
          dj(halo_size, halo_size, halo_size, d2 - halo_size - 1, d2), grid(di, dj),
#if FLOAT_PRECISION == 4
          verifier_(1e-6),
#else
          verifier_(1e-12),
#endif
          verifier_halos{{{halo_size, halo_size}, {halo_size, halo_size}, {halo_size, halo_size}}},
          in_1(meta_, 1., "in_1"),                //
          in_2(meta_, 2., "in_2"),                //
          in_3(meta_, 3., "in_3"),                //
          in_4(meta_, 4., "in_4"),                //
          in_5(meta_, 5., "in_5"),                //
          out_1(meta_, -1., "out_1"),             //
          out_2(meta_, -2., "out_2"),             //
          out_3(meta_, -3., "out_3"),             //
          out_4(meta_, -4., "out_4"),             //
          out_5(meta_, -5., "out_5"),             //
          in{in_1, in_2, in_3, in_4, in_5},       //
          out{out_1, out_2, out_3, out_4, out_5}, //
          domain(in, out) {
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;
    }

    template < typename Computation >
    void execute_computation(Computation &comp) {
        comp->ready();
        comp->steady();
        comp->run();
        out_1.sync();
        out_2.sync();
        out_3.sync();
        out_4.sync();
        out_5.sync();
    }
};

TEST_F(expandable_parameters, copy) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(expand_factor< 2 >(),
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(), gridtools::make_stage< copy_functor >(p_out(), p_in())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, in_1, out_1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_2, out_2, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_3, out_3, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_4, out_4, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_5, out_5, verifier_halos));
}

TEST_F(expandable_parameters, call_proc_copy) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(
        expand_factor< 2 >(),
        domain,
        grid,
        gridtools::make_multistage(
            execute< forward >(), gridtools::make_stage< call_proc_copy_functor >(p_out(), p_in())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, in_1, out_1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_2, out_2, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_3, out_3, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_4, out_4, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_5, out_5, verifier_halos));
}

TEST_F(expandable_parameters, call_copy) {
    auto comp = gridtools::make_computation< gridtools::BACKEND >(expand_factor< 2 >(),
        domain,
        grid,
        gridtools::make_multistage(execute< forward >(), gridtools::make_stage< call_copy_functor >(p_out(), p_in())));

    execute_computation(comp);

    ASSERT_TRUE(verifier_.verify(grid, in_1, out_1, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_2, out_2, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_3, out_3, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_4, out_4, verifier_halos));
    ASSERT_TRUE(verifier_.verify(grid, in_5, out_5, verifier_halos));
}
