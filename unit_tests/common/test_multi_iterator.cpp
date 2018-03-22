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

#include "common/multi_iterator.hpp"
#include <common/tuple.hpp>
#include "common/pair.hpp"
#include <vector>
#include "../tools/multiplet.hpp"
#include <gtest/gtest.h>

using namespace gridtools;

template < typename T >
void print(array< T, 3 > a) {
    std::cout << a << std::endl;
}

const range i_range = {1, 3};
const range j_range = {4, 8};
const range k_range = {2, 10};
const size_t size =
    (i_range.end() - i_range.begin()) * (j_range.end() - j_range.begin()) * (k_range.end() - k_range.begin());

namespace {
    void verify(const std::vector< multiplet< 3 > > &out) {
        ASSERT_EQ(size, out.size()) << " Number of iterated elements is incorrect.";
        size_t count = 0;
        for (size_t i = i_range.begin(); i < i_range.end(); ++i)
            for (size_t j = j_range.begin(); j < j_range.end(); ++j)
                for (size_t k = k_range.begin(); k < k_range.end(); ++k) {
                    EXPECT_EQ((multiplet< 3 >{i, j, k}), out[count]);
                    count++;
                }
    }
}

TEST(test_hypercube_view, explicit_hypercube_instantiation) {
    std::vector< multiplet< 3 > > out;

    hypercube_view< 3 > view(hypercube< 3 >{i_range, j_range, k_range});
    for (auto it : view) {
        out.emplace_back(make_multiplet(it[0], it[1], it[2]));
    }

    verify(out);
}

// make_hypercube_view with ranges does not work with CUDA9.1 and earlier
#ifndef __CUDACC__
TEST(test_hypercube_view, make_hypercube_view_from_ranges) {
    std::vector< multiplet< 3 > > out;

    auto view = make_hypercube_view(i_range, j_range, k_range);
    for (auto it : view) {
        out.emplace_back(make_multiplet(it[0], it[1], it[2]));
    }

    verify(out);
}

// TEST(test_hypercube_view, make_hypercube_view_from_tuples) {
//    std::vector< multiplet< 3 > > out;
//
//    auto view = make_hypercube_view(make_tuple(i_range.begin(), i_range.end()),
//        make_tuple(j_range.begin(), j_range.end()),
//        make_tuple(k_range.begin(), k_range.end()));
//    for (auto it : view) {
//        out.emplace_back(make_multiplet(it[0], it[1], it[2]));
//    }
//
//    verify(out);
//}

TEST(test_hypercube_view, make_hypercube_view_from_pairs) {
    std::vector< multiplet< 3 > > out;

    auto view = make_hypercube_view(make_pair(i_range.begin(), i_range.end()),
        make_pair(j_range.begin(), j_range.end()),
        make_pair(k_range.begin(), k_range.end()));
    for (auto it : view) {
        out.emplace_back(make_multiplet(it[0], it[1], it[2]));
    }

    verify(out);
}

// TEST(test_hypercube_view_iterate_from_zero, from_sizes) {
//    size_t size_i = 3;
//    size_t size_j = 4;
//    size_t size_k = 5;
//    hypercube_view< 3 > expect(hypercube< 3 >{range(0, size_i), range(0, size_j), range(0, size_k)});
//
//    auto view = make_hypercube_view(array< size_t, 3 >{size_i, size_j, size_k});
//
//    ASSERT_EQ(expect, view);
//}
#endif

TEST(test_hypercube_view, make_hypercube_view_from_hypercube) {
    std::vector< multiplet< 3 > > out;

    auto view = make_hypercube_view(hypercube< 3 >{i_range, j_range, k_range});
    for (auto it : view) {
        out.emplace_back(make_multiplet(it[0], it[1], it[2]));
    }

    verify(out);
}
