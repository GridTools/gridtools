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
#include <distributed-boundaries/binded_bc.hpp>

using namespace std::placeholders;
namespace gt = gridtools;

TEST(DistributedBoundaries, SelectElement) {
    auto all = std::make_tuple(1, _1, 3, _2);
    auto sub = std::make_tuple(2, 4);

    EXPECT_EQ(gt::_impl::select_element< 0 >(sub, all, gt::_impl::NotPlc{}), 1);
    EXPECT_EQ(gt::_impl::select_element< 1 >(sub, all, gt::_impl::Plc{}), 2);
    EXPECT_EQ(gt::_impl::select_element< 2 >(sub, all, gt::_impl::NotPlc{}), 3);
    EXPECT_EQ(gt::_impl::select_element< 3 >(sub, all, gt::_impl::Plc{}), 4);
}

TEST(DistributedBoundaries, SubstitutePlaceholders) {
    auto all = std::make_tuple(1, _1, 3, _2);
    auto sub = std::make_tuple(2, 4);

    auto res = gt::_impl::substitute_placeholders(
        sub, all, typename gt::make_gt_integer_sequence< gt::uint_t, std::tuple_size< decltype(all) >::value >::type{});

    EXPECT_EQ(res, (std::tuple< int, int, int, int >(1, 2, 3, 4)));
}

TEST(DistributedBoundaries, RestTuple) {
    auto tup = std::make_tuple(1, 2, 3, 4, 5);

    auto rest = gt::_impl::rest_tuple(
        tup, typename gt::make_gt_integer_sequence< gt::uint_t, std::tuple_size< decltype(tup) >::value - 1 >::type{});

    EXPECT_EQ(rest, (std::tuple< int, int, int, int >(2, 3, 4, 5)));
}

TEST(DistributedBoundaries, RemovePlaceholders) {
    {
        auto all = std::make_tuple();

        auto res = gt::_impl::remove_placeholders(all);

        EXPECT_EQ(res, (std::tuple<>{}));
    }
    {
        auto all = std::make_tuple(1);

        auto res = gt::_impl::remove_placeholders(all);

        EXPECT_EQ(res, (std::tuple< int >{1}));
    }
    {
        auto all = std::make_tuple(1, 2);

        auto res = gt::_impl::remove_placeholders(all);

        EXPECT_EQ(res, (std::tuple< int, int >{1, 2}));
    }
    {
        auto all = std::make_tuple(_1);

        auto res = gt::_impl::remove_placeholders(all);

        EXPECT_EQ(res, (std::tuple<>{}));
    }
    {
        auto all = std::make_tuple(1, _1);

        auto res = gt::_impl::remove_placeholders(all);

        EXPECT_EQ(res, (std::tuple< int >{1}));
    }
}
