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

using index_pair = std::pair<int, int>;
using index_pair_vector = std::vector<std::pair<index_pair, index_pair>>;

// helper printing the loop index pairs
struct GetIndexPairs {
    GetIndexPairs(index_pair_vector &index_pairs) : index_pairs(index_pairs) {}

    template <typename TIndexPair>
    void operator()(TIndexPair) {
        // extract the level information
        typedef typename index_to_level<typename boost::mpl::first<TIndexPair>::type>::type FromLevel;
        typedef typename index_to_level<typename boost::mpl::second<TIndexPair>::type>::type ToLevel;
        static constexpr int from_splitter = FromLevel::splitter;
        static constexpr int from_offset = FromLevel::offset;
        static constexpr int to_splitter = ToLevel::splitter;
        static constexpr int to_offset = ToLevel::offset;

        index_pairs.emplace_back(index_pair(from_splitter, from_offset), index_pair(to_splitter, to_offset));
    }

    index_pair_vector &index_pairs;
};

// test method computing loop intervals
TEST(test_loop_intervals, test_compute_functor_do_methods) {
    // define the axis search interval
    typedef interval<level_t<0, -3>, level_t<3, 3>> AxisInterval;

    // compute the functor do methods
    typedef boost::mpl::transform<boost::mpl::vector<Functor0, Functor1, Functor2>,
        compute_functor_do_methods<boost::mpl::_, AxisInterval>>::type FunctorDoMethods;

    index_pair_vector index_pairs;
    boost::mpl::for_each<compute_loop_intervals<FunctorDoMethods, AxisInterval>::type>(GetIndexPairs(index_pairs));

    // verfify intervals
    ASSERT_EQ(index_pairs.size(), 4);
    EXPECT_EQ(index_pairs.at(0).first, index_pair(0, 1));
    EXPECT_EQ(index_pairs.at(0).second, index_pair(1, -1));
    EXPECT_EQ(index_pairs.at(1).first, index_pair(1, 1));
    EXPECT_EQ(index_pairs.at(1).second, index_pair(2, -1));
    EXPECT_EQ(index_pairs.at(2).first, index_pair(2, 1));
    EXPECT_EQ(index_pairs.at(2).second, index_pair(3, -2));
    EXPECT_EQ(index_pairs.at(3).first, index_pair(3, -1));
    EXPECT_EQ(index_pairs.at(3).second, index_pair(3, -1));
}
