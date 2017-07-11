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

#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include <gridtools.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "gtest/gtest.h"

#include <stencil-composition/stencil-composition.hpp>
#include "stencil-composition/backend.hpp"
#include "stencil-composition/make_computation.hpp"
#include "stencil-composition/make_stencils.hpp"

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
#define BACKEND_ARCH Cuda
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#define BACKEND_ARCH Host
#define BACKEND backend< Host, GRIDBACKEND, Block >
#endif

namespace make_computation_test {

    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    struct test_functor {
        typedef accessor< 0, inout > inacc;
        typedef boost::mpl::vector1< inacc > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval) {}
    };

    struct test_functor_two_x_ext {
        typedef accessor< 0, in, extent< -2, 2, 0, 0 > > inacc;
        typedef accessor< 1, inout > outacc;
        typedef boost::mpl::vector2< inacc, outacc > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval) {}
    };

    struct test_functor_two_y_ext {
        typedef accessor< 0, in, extent< 0, 0, -2, 2 > > inacc;
        typedef accessor< 1, inout > outacc;
        typedef boost::mpl::vector2< inacc, outacc > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval) {}
    };
}

TEST(MakeComputation, Basic) {
    typedef storage_traits< BACKEND_ARCH >::storage_info_t< 0, 3 > storage_info_t;
    typedef storage_traits< BACKEND_ARCH >::data_store_t< float_type, storage_info_t > data_store_t;

    storage_info_t meta_data_(4, 4, 4);
    data_store_t in(meta_data_, [](int i, int j, int k) { return i + j + k; }, "in");

    typedef arg< 0, data_store_t > p_in;
    typedef boost::mpl::vector< p_in > accessor_list;

    gridtools::aggregator_type< accessor_list > domain((p_in() = in));

    uint_t di[5] = {0, 0, 0, 4 - 1, 4};
    uint_t dj[5] = {0, 0, 0, 4 - 1, 4};

    gridtools::grid< make_computation_test::axis > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = 4 - 1;

    // create the computation, check that the make_computation call succeeds
    auto test = gridtools::make_computation< gridtools::BACKEND >(domain,
        grid,
        gridtools::make_multistage // mss_descriptor
        (execute< forward >(), gridtools::make_stage< make_computation_test::test_functor >(p_in())));
}

TEST(MakeComputation, InvalidHalo) {
    typedef storage_traits< BACKEND_ARCH >::storage_info_t< 0, 3 > storage_info_t;
    typedef storage_traits< BACKEND_ARCH >::data_store_t< float_type, storage_info_t > data_store_t;

    storage_info_t meta_data_(4, 4, 4);
    data_store_t in(meta_data_, [](int i, int j, int k) { return i + j + k; }, "in");
    data_store_t out(meta_data_, 0., "out");

    typedef arg< 0, data_store_t > p_in;
    typedef arg< 1, data_store_t > p_out;

    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    gridtools::aggregator_type< accessor_list > domain(in, out);
    constexpr int halo_size = 1;

    uint_t di[5] = {halo_size, halo_size, halo_size, 4 - 1 - halo_size, 4};
    uint_t dj[5] = {halo_size, halo_size, halo_size, 4 - 1 - halo_size, 4};

    gridtools::grid< make_computation_test::axis > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = 4 - 1;

#ifndef NDEBUG
    // create the computation, check that the make_computation call fails because the halo in x in the
    // test_functor_two_x_ext is 2.
    ASSERT_DEATH((gridtools::make_computation< gridtools::BACKEND >(
                     domain,
                     grid,
                     gridtools::make_multistage // mss_descriptor
                     (execute< forward >(),
                         gridtools::make_stage< make_computation_test::test_functor_two_x_ext >(p_in(), p_out())))),
        "One of the stencil accessor extents is exceeding the halo region.");
    // create the computation, check that the make_computation call fails because the halo in y in the
    // test_functor_two_y_ext is 2.
    ASSERT_DEATH((gridtools::make_computation< gridtools::BACKEND >(
                     domain,
                     grid,
                     gridtools::make_multistage // mss_descriptor
                     (execute< forward >(),
                         gridtools::make_stage< make_computation_test::test_functor_two_y_ext >(p_in(), p_out())))),
        "One of the stencil accessor extents is exceeding the halo region.");
#endif
}
