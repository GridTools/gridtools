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

using namespace gridtools;
using namespace enumtype;


#ifdef __CUDACC__
#define BACKEND_ARCH Cuda
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#define BACKEND_ARCH Host
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif


// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, -1 > > x_interval;
typedef gridtools::interval< gridtools::level< 0, -2 >, gridtools::level< 1, 1 > > axis;

typedef storage_traits< BACKEND_ARCH >::storage_info_t< 0, 3 > meta_t;
typedef storage_traits< BACKEND_ARCH >::data_store_t< float_type, meta_t > storage_t;

typedef arg< 0, storage_t > p_in;
typedef arg< 1, storage_t > p_out;

template <int I, int J, int K, typename Extent>
struct functor {
    typedef accessor< 0, enumtype::in, Extent, 3 > in;
    typedef accessor< 1, enumtype::inout, extent< >, 3 > out;
    typedef boost::mpl::vector< in, out > arg_list;

    template < typename Evaluation >
    GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
        eval(out()) = eval(in(I,J,K));
    }
};

// first check 
template < int I, int J, int K, typename Extent, typename Domain, typename Grid >
void test_with(Domain domain, Grid grid) {
    auto test = gridtools::make_computation< gridtools::BACKEND >(domain, grid,
        gridtools::make_multistage // mss_descriptor
        (execute< forward >(), gridtools::make_stage< functor< I, J, K, Extent > >(p_in(), p_out())));

    test->ready();
    test->steady();
    test->run();
}

TEST(Accessor, OutOfBounds) {
    uint_t d1 = 15;
    uint_t d2 = 13;
    uint_t d3 = 18;

    meta_t si(d1, d2, d3);
    storage_t in(si);
    storage_t out(si);

    typedef boost::mpl::vector< p_in, p_out > accessor_list;

    gridtools::aggregator_type< accessor_list > domain((p_in() = in), (p_out() = out));

    uint_t di[5] = {3, 3, 3, d1 - 4, d1};
    uint_t dj[5] = {3, 3, 3, d2 - 4, d2};

    gridtools::grid< axis > grid(di, dj);
    grid.value_list[0] = 2;
    grid.value_list[1] = d3 - 2;

#ifndef NDEBUG
    // positive test cases
    // all zero check
    EXPECT_NO_THROW((test_with<0,0,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // one minus check
    EXPECT_NO_THROW((test_with<-1,0,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_NO_THROW((test_with<0,-1,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_NO_THROW((test_with<0,0,-1, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // two minus check
    EXPECT_NO_THROW((test_with<-1,-1,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_NO_THROW((test_with<0,-1,-1, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_NO_THROW((test_with<-1,0,-1, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // all minus check
    EXPECT_NO_THROW((test_with<-1,-1,-1, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // one plus check
    EXPECT_NO_THROW((test_with<1,0,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_NO_THROW((test_with<0,1,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_NO_THROW((test_with<0,0,1, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // two plus check
    EXPECT_NO_THROW((test_with<1,1,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_NO_THROW((test_with<0,1,1, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_NO_THROW((test_with<1,0,1, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // all plus check
    EXPECT_NO_THROW((test_with<1,1,1, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // mixed
    EXPECT_NO_THROW((test_with<-1,1,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_NO_THROW((test_with<0,1,-1, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_NO_THROW((test_with<1,0,-1, extent<-1,1,-1,1,-1,1> >(domain, grid)));

    // negative test cases
    // one minus check
    EXPECT_ANY_THROW((test_with<-2,0,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_ANY_THROW((test_with<0,-3,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_ANY_THROW((test_with<0,0,-5, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // two minus check
    EXPECT_ANY_THROW((test_with<-2,-5,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_ANY_THROW((test_with<0,-1,-2, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_ANY_THROW((test_with<-6,0,-1, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // all minus check
    EXPECT_ANY_THROW((test_with<-3,-2,-1, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // one plus check
    EXPECT_ANY_THROW((test_with<2,0,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_ANY_THROW((test_with<0,3,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_ANY_THROW((test_with<0,0,5, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // two plus check
    EXPECT_ANY_THROW((test_with<1,2,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_ANY_THROW((test_with<0,2,3, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_ANY_THROW((test_with<1,0,5, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // all plus check
    EXPECT_ANY_THROW((test_with<5,4,2, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    // mixed
    EXPECT_ANY_THROW((test_with<-1,2,0, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_ANY_THROW((test_with<0,5,-3, extent<-1,1,-1,1,-1,1> >(domain, grid)));
    EXPECT_ANY_THROW((test_with<1,0,-2, extent<-1,1,-1,1,-1,1> >(domain, grid)));
#endif
}

