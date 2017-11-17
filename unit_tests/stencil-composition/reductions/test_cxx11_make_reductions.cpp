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
/*
 * test_computation.cpp
 *
 *  Created on: Mar 9, 2015
 *      Author: carlosos
 */

#include <gridtools.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "gtest/gtest.h"

#include <stencil-composition/stencil-composition.hpp>
#include "stencil-composition/backend.hpp"
#include "stencil-composition/make_computation.hpp"
#include "stencil-composition/make_stencils.hpp"
#include "stencil-composition/reductions/reductions.hpp"

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#elif defined(__AVX512F__)
#define BACKEND backend< Mic, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

namespace make_reduction_test {

    using x_interval = gridtools::interval< level< 0, -1 >, level< 1, -1 > >;

    struct test_functor {
        typedef accessor< 0 > in;
        typedef boost::mpl::vector1< in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {}
    };
}

TEST(test_make_reduction, make_reduction) {

    using namespace gridtools;

    typedef BACKEND::storage_traits_t::storage_info_t< 0, 6 > storage_info_t;
    typedef BACKEND::storage_traits_t::data_store_t< float_type, storage_info_t > storage_t;

    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_t > p_out;
    typedef boost::mpl::vector< p_in, p_out > accessor_list_t;

    typedef decltype(gridtools::make_reduction< make_reduction_test::test_functor, binop::sum >(0.0, p_in())) red_t;

    typedef reduction_descriptor< double,
        binop::sum,
        boost::mpl::vector1< esf_descriptor< make_reduction_test::test_functor, boost::mpl::vector1< p_in > > > >
        red_ref_t;

    GRIDTOOLS_STATIC_ASSERT((red_t::is_reduction_t::value), "ERROR");
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal< red_t::esf_sequence_t,
                                red_ref_t::esf_sequence_t,
                                esf_equal< boost::mpl::_1, boost::mpl::_2 > >::type::value),
        "ERROR");

    EXPECT_TRUE(true);
}
