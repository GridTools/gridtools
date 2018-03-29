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

#include <common/hypercube_iterator.hpp>
#include <common/tuple.hpp>
#include "common/pair.hpp"
#include <vector>
#include "../tools/multiplet.hpp"
#include <gtest/gtest.h>

using namespace gridtools;

class test_hypercube_view_fixture : public ::testing::Test {
  public:
    test_hypercube_view_fixture(
        pair< size_t, size_t > i_range, pair< size_t, size_t > j_range, pair< size_t, size_t > k_range)
        : i_range(i_range), j_range(j_range), k_range(k_range),
          size((i_range.second - i_range.first) * (j_range.second - j_range.first) * (k_range.second - k_range.first)) {
    }

    const pair< size_t, size_t > i_range;
    const pair< size_t, size_t > j_range;
    const pair< size_t, size_t > k_range;
    const size_t size;

    void verify(const std::vector< multiplet< 3 > > &out) {
        ASSERT_EQ(size, out.size()) << " Number of iterated elements is incorrect.";
        size_t count = 0;
        for (size_t i = i_range.first; i < i_range.second; ++i)
            for (size_t j = j_range.first; j < j_range.second; ++j)
                for (size_t k = k_range.first; k < k_range.second; ++k) {
                    EXPECT_EQ((multiplet< 3 >{i, j, k}), out[count]);
                    count++;
                }
    }
};

class test_hypercube_view : public test_hypercube_view_fixture {
  public:
    test_hypercube_view() : test_hypercube_view_fixture({1, 3}, {4, 8}, {2, 10}) {}
};

TEST_F(test_hypercube_view, make_hypercube_view_from_array_of_ranges) {
    std::vector< multiplet< 3 > > out;

    gridtools::array< gridtools::pair< size_t, size_t >, 3 > cube{i_range, j_range, k_range};
    auto view = make_hypercube_view(cube);
    for (auto it : view) {
        out.emplace_back(make_multiplet(it[0], it[1], it[2]));
    }

    verify(out);
}

// TODO enable once tuple is more std-compliant
// TEST_F(test_hypercube_view, make_hypercube_view_from_tuple_of_ranges) {
//    std::vector< multiplet< 3 > > out;
//
//    gridtools::tuple< gridtools::pair< size_t, size_t >,
//        gridtools::pair< size_t, size_t >,
//        gridtools::pair< size_t, size_t > > cube{i_range, j_range, k_range};
//    auto view = make_hypercube_view(cube);
//    for (auto it : view) {
//        out.emplace_back(make_multiplet(it[0], it[1], it[2]));
//    }
//
//    verify(out);
//}

class test_hypercube_view_from_zero : public test_hypercube_view_fixture {
  public:
    test_hypercube_view_from_zero() : test_hypercube_view_fixture({0, 3}, {0, 4}, {0, 5}) {}
};

TEST_F(test_hypercube_view_from_zero, from_array_of_integers) {
    std::vector< multiplet< 3 > > out;

    auto view = make_hypercube_view_from_zero(array< size_t, 3 >{i_range.second, j_range.second, k_range.second});
    for (auto it : view) {
        out.emplace_back(make_multiplet(it[0], it[1], it[2]));
    }

    verify(out);
}

TEST(test_hypercube_view_empty_iteration_space, from_zero_to_zero) {
    std::vector< multiplet< 3 > > out;

    auto view = make_hypercube_view_from_zero(array< size_t, 3 >{0, 0, 0});
    for (auto it : view) {
        out.emplace_back(make_multiplet(it[0], it[1], it[2]));
    }

    ASSERT_EQ(0, out.size());
}

TEST(test_hypercube_view_empty_iteration_space, from_one_to_one) {
    std::vector< multiplet< 3 > > out;

    auto view = make_hypercube_view(
        gridtools::array< gridtools::pair< size_t, size_t >, 3 >{gridtools::pair< size_t, size_t >{1, 1},
            gridtools::pair< size_t, size_t >{1, 1},
            gridtools::pair< size_t, size_t >{1, 1}});
    for (auto it : view) {
        out.emplace_back(make_multiplet(it[0], it[1], it[2]));
    }

    ASSERT_EQ(0, out.size());
}

TEST(test_hypercube_view_empty_iteration_space, zero_dimensional) {
    std::vector< multiplet< 3 > > out;

    auto view = make_hypercube_view(gridtools::array< gridtools::pair< size_t, size_t >, 0 >{});
    for (auto it : view) {
        out.emplace_back(make_multiplet(it[0], it[1], it[2]));
    }

    ASSERT_EQ(0, out.size());
}
