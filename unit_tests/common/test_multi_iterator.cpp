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

#include <gtest/gtest.h>
#include "common/multi_iterator.hpp"
#include <vector>
#include "../tools/multiplet.hpp"

using namespace gridtools;

template < typename T >
void print(array< T, 3 > a) {
    std::cout << a << std::endl;
}

class test_hypercube_view : public testing::Test {
  public:
    const range i_range = {1, 3};
    const range j_range = {4, 8};
    const range k_range = {2, 10};

    const size_t size =
        (i_range.begin() - i_range.end()) + (j_range.begin() - j_range.end()) + (j_range.begin() - j_range.end());
};

TEST_F(test_hypercube_view, iteration) {
    std::vector< multiplet< 3 > > out;

    hypercube_view< 3 > view(hypercube< 3 >{i_range, j_range, k_range});
    for (auto it : view) {
        out.emplace_back(it[0], it[1], it[2]);
    }

    size_t count = 0;
    for (size_t i = i_range.begin(); i < i_range.end(); ++i)
        for (size_t j = j_range.begin(); j < j_range.end(); ++j)
            for (size_t k = k_range.begin(); k < k_range.end(); ++k) {
                ASSERT_EQ((multiplet< 3 >{i, j, k}), out[count]);
                count++;
            }
    ASSERT_EQ(count, out.size()) << " iterated over too many elements";
}

TEST_F(test_hypercube_view, make_hypercube_view_from_ranges) {
    hypercube_view< 3 > expect(hypercube< 3 >{i_range, j_range, k_range});

    auto view = make_hypercube_view(i_range, j_range, k_range);

    ASSERT_EQ(expect, view);
}

TEST_F(test_hypercube_view, make_hypercube_view_from_hypercube) {
    hypercube_view< 3 > expect(hypercube< 3 >{i_range, j_range, k_range});

    auto view = make_hypercube_view(hypercube< 3 >{i_range, j_range, k_range});

    ASSERT_EQ(expect, view);
}
