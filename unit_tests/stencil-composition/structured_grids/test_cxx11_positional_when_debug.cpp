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

#ifdef NDEBUG
#undef NDEBUG
#define __WAS_DEBUG
#endif

#include <gridtools.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "gtest/gtest.h"

#include <stencil-composition/stencil-composition.hpp>
#include "stencil-composition/backend.hpp"
#include "stencil-composition/make_computation.hpp"
#include "stencil-composition/make_stencils.hpp"
#include "backend_select.hpp"

namespace positional_when_debug_test {

    using grid_t = gridtools::grid< gridtools::axis< 1 >::axis_interval_t >;
    typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, -1 > > x_interval;
    typedef gridtools::interval< gridtools::level< 0, -2 >, gridtools::level< 1, 1 > > axis_t;

    struct test_functor {
        typedef gridtools::accessor< 0, gridtools::enumtype::inout > in;
        typedef boost::mpl::vector1< in > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval.i();
            eval.j();
            eval.k();
        }
    };
}

TEST(test_make_computation, positional_when_debug) {

    using namespace gridtools;
    using namespace gridtools::enumtype;

    typedef backend_t::storage_traits_t::storage_info_t< 0, 3 > meta_data_t;
    typedef backend_t::storage_traits_t::data_store_t< float_type, meta_data_t > storage_t;

    typedef arg< 0, storage_t > p_in;

    make_computation< backend_t >(positional_when_debug_test::grid_t(halo_descriptor{}, halo_descriptor{}, {0, 0}),
        make_multistage // mss_descriptor
        (execute< forward >(), make_stage< positional_when_debug_test::test_functor >(p_in())));
}

#ifdef __WAS_DEBUG
#undef __WAS_DEBUG
#define NDEBUG
#endif
