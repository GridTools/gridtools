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

#include <common/pair.hpp>
#include <common/make_array.hpp>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace gridtools;

class hypercube_iteration : public ::testing::Test {
    using range_t = std::array< size_t, 2 >;
    using data_t = std::vector< std::vector< size_t > >;

  public:
    void expect_ranges(range_t range_i, range_t range_j, range_t range_k) {
        for (size_t i = range_i[0]; i != range_i[1]; ++i)
            for (size_t j = range_j[0]; j != range_j[1]; ++j)
                for (size_t k = range_k[0]; k != range_k[1]; ++k)
                    m_expected.push_back({i, j, k});
    }

    template < class R >
    void call_testee(R &&range) {
        for (auto &&item : make_hypercube_view(std::forward< R >(range)))
            m_actual.push_back({item[0], item[1], item[2]});
    }

    ~hypercube_iteration() { EXPECT_THAT(m_actual, testing::ContainerEq(m_expected)); }

  private:
    data_t m_actual;
    data_t m_expected;
};

TEST_F(hypercube_iteration, view_from_array_of_ranges) {
    expect_ranges({1, 3}, {4, 8}, {2, 10});
    call_testee(make_array(make_pair(1, 3), make_pair(4, 8), make_pair(2, 10)));
}

// TODO enable once tuple is more std-compliant
// TEST_F(hypercube_iteration, view_from_tuple_of_ranges) {
//    expect_ranges({1, 3}, {4, 8}, {2, 10});
//    call_testee(make_tuple(make_pair(1, 3), make_pair(4, 8), make_pair(2, 10)));
//}

TEST_F(hypercube_iteration, from_array_of_integers) {
    expect_ranges({0, 3}, {0, 8}, {0, 10});
    call_testee(make_array(3, 8, 10));
}

TEST_F(hypercube_iteration, from_zero_to_zero) {
    expect_ranges({}, {}, {});
    call_testee(make_array(make_pair(0, 0), make_pair(0, 0), make_pair(0, 0)));
}

TEST_F(hypercube_iteration, from_one_to_one) {
    expect_ranges({}, {}, {});
    call_testee(make_array(0, 0, 0));
}

TEST(hypercube_view_empty_iteration_space, zero_dimensional_range) {
    auto view = make_hypercube_view(array< pair< size_t, size_t >, 0 >{});
    for (auto it : view)
        FAIL();
}

TEST(hypercube_view_empty_iteration_space, zero_dimensional_size) {
    auto view = make_hypercube_view(array< size_t, 0 >{});
    for (auto it : view)
        FAIL();
}
