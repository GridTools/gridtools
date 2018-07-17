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
#include "test_helper.hpp"
#include "gtest/gtest.h"

#include <boost/mpl/for_each.hpp>
#include <boost/mpl/transform.hpp>
#include <gridtools/stencil-composition/functor_do_methods.hpp>
#include <gridtools/stencil-composition/interval.hpp>
#include <gridtools/stencil-composition/loopintervals.hpp>
#include <iostream>
#include <vector>

using namespace gridtools;

constexpr int level_offset_limit = 3;

template <uint_t Splitter, int_t Offset>
using level_t = level<Splitter, Offset, level_offset_limit>;

// test functor 1
struct Functor0 {
    template <typename TArguments>
    static void Do(TArguments &args, interval<level_t<3, -1>, level_t<3, -1>>) {}
};

// test functor 1
struct Functor1 {
    template <typename TArguments>
    static void Do(TArguments &args, interval<level_t<0, 1>, level_t<2, -1>>) {}
};

// test functor 2
struct Functor2 {
    template <typename TArguments>
    static void Do(TArguments &args, interval<level_t<0, 1>, level_t<1, -1>>) {}

    template <typename TArguments>
    static void Do(TArguments &args, interval<level_t<1, 1>, level_t<3, -1>>) {}
};

// test method computing loop intervals
TEST(test_loop_intervals, test_compute_functor_do_methods) {
    // define the axis search interval
    typedef interval<level_t<0, -3>, level_t<3, 3>> AxisInterval;

    // compute the functor do methods
    typedef boost::mpl::transform<boost::mpl::vector<Functor0, Functor1, Functor2>,
        compute_functor_do_methods<boost::mpl::_, AxisInterval>>::type FunctorDoMethods;

    using loop_intervals = typename compute_loop_intervals<FunctorDoMethods, AxisInterval>::type;

    using interval0 = typename boost::mpl::at_c<loop_intervals, 0>::type;
    using from0 = typename index_to_level<typename interval0::first>::type;
    using to0 = typename index_to_level<typename interval0::second>::type;
    ASSERT_TYPE_EQ<from0, level_t<0, 1>>();
    ASSERT_TYPE_EQ<to0, level_t<1, -1>>();

    using interval1 = typename boost::mpl::at_c<loop_intervals, 1>::type;
    using from1 = typename index_to_level<typename interval1::first>::type;
    using to1 = typename index_to_level<typename interval1::second>::type;
    ASSERT_TYPE_EQ<from1, level_t<1, 1>>();
    ASSERT_TYPE_EQ<to1, level_t<2, -1>>();

    using interval2 = typename boost::mpl::at_c<loop_intervals, 2>::type;
    using from2 = typename index_to_level<typename interval2::first>::type;
    using to2 = typename index_to_level<typename interval2::second>::type;
    ASSERT_TYPE_EQ<from2, level_t<2, 1>>();
    ASSERT_TYPE_EQ<to2, level_t<3, -2>>();

    using interval3 = typename boost::mpl::at_c<loop_intervals, 3>::type;
    using from3 = typename index_to_level<typename interval3::first>::type;
    using to3 = typename index_to_level<typename interval3::second>::type;
    ASSERT_TYPE_EQ<from3, level_t<3, -1>>();
    ASSERT_TYPE_EQ<to3, level_t<3, -1>>();
}
