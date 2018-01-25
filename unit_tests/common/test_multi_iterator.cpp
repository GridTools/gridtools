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
#include "common/array.hpp"
#include "common/make_array.hpp"
#include "../tools/multiplet.hpp"

using namespace gridtools;

using range_t = pair< uint_t, uint >;

// TEST(make_multi_iterator, range) {
//    const uint_t i_begin = 1;
//    const uint_t i_end = 3;
//    const uint_t j_begin = 4;
//    const uint_t j_end = 8;
//
//    auto range = gridtools::make_array(range_t(i_begin, i_end), range_t(j_begin, j_end));
//    multi_iterator< uint_t, 2 > expected(range);
//
//    ASSERT_EQ(expected, make_multi_iterator(range)); // from array
//    ASSERT_EQ(
//        expected, make_multi_iterator(range_t{i_begin, i_end}, range_t{j_begin, j_end})); // from sequence of pairs
//    ASSERT_EQ(expected, make_multi_iterator({i_begin, i_end}, {j_begin, j_end})); // from braced initializer list
//}
//
// TEST(make_multi_iterator, from_zero) {
//    const uint_t i_begin = 0;
//    const uint_t i_end = 3;
//    const uint_t j_begin = 0;
//    const uint_t j_end = 8;
//
//    auto range = gridtools::make_array(range_t(i_begin, i_end), range_t(j_begin, j_end));
//    multi_iterator< uint_t, 2 > expected(range);
//
//    ASSERT_EQ(expected, make_multi_iterator(make_array(i_end, j_end))); // from array of endpoints (not pair)
//    ASSERT_EQ(expected, make_multi_iterator(i_end, j_end));             // from sequence of end points
//}
//
// class test_multi_iterator : public testing::Test {
//  public:
//    const range_t i_range = {1, 3};
//    const range_t j_range = {4, 8};
//    const range_t k_range = {2, 10};
//
//    const size_t size =
//        (i_range.second - i_range.first) + (j_range.second - j_range.first) + (k_range.second - k_range.first);
//};
//
// TEST_F(test_multi_iterator, 3D) {
//    std::vector< multiplet< 3 > > out;
//
//    make_multi_iterator(i_range, j_range, k_range)
//        .iterate([&](size_t a, size_t b, size_t c) {
//            out.push_back(multiplet< 3 >{a, b, c});
//        });
//
//    int count = 0;
//    for (size_t i = i_range.first; i < i_range.second; ++i)
//        for (size_t j = j_range.first; j < j_range.second; ++j)
//            for (size_t k = k_range.first; k < k_range.second; ++k) {
//                ASSERT_EQ(1, std::count(out.begin(), out.end(), multiplet< 3 >{i, j, k})); // all elements are there
//                count++;
//            }
//    ASSERT_EQ(count, out.size()); // no extra elements are there
//}
//
// TEST_F(test_multi_iterator, 3D_reduction) {
//    // just count
//    const size_t expected =
//        (i_range.second - i_range.first) * (j_range.second - j_range.first) * (k_range.second - k_range.first);
//
//    auto actual =
//        make_multi_iterator(i_range, j_range, k_range)
//            .reduce([&](size_t a, size_t b, size_t c) { return 1; }, [](size_t a, size_t b) { return a + b; }, 0);
//
//    ASSERT_EQ(expected, actual);
//}
//
// TEST_F(test_multi_iterator, 2D) {
//    std::vector< multiplet< 2 > > out;
//
//    make_multi_iterator(i_range, j_range).iterate([&](size_t a, size_t b) { out.push_back(multiplet< 2 >{a, b}); });
//
//    int count = 0;
//    for (size_t i = i_range.first; i < i_range.second; ++i)
//        for (size_t j = j_range.first; j < j_range.second; ++j) {
//            ASSERT_EQ(1, std::count(out.begin(), out.end(), multiplet< 2 >{i, j}));
//            count++;
//        }
//    ASSERT_EQ(count, out.size());
//}
//
// TEST(test_multi_iterator_empty, 0D_array) {
//    gridtools::array< gridtools::pair< uint_t, uint_t >, 0 > dims;
//    std::vector< int > out;
//
//    make_multi_iterator(dims).iterate([&]() { out.push_back(0); });
//
//    ASSERT_EQ(0, out.size());
//}
//
// TEST(test_multi_iterator_empty, 0D) {
//    std::vector< int > out;
//
//    make_multi_iterator().iterate([&]() { out.push_back(0); });
//
//    ASSERT_EQ(0, out.size());
//}
//
// TEST(test_multi_iterator_empty, 0D_reduction) {
//    const size_t init_val = 3;
//
//    const size_t expected = init_val;
//
//    auto actual = make_multi_iterator().reduce([&]() { return 0; }, [](size_t a, size_t b) { return 0; }, init_val);
//
//    ASSERT_EQ(expected, actual);
//}
//
// TEST(test_multi_iterator_empty, 2D) {
//    using dim2_ = gridtools::array< size_t, 2 >;
//    std::vector< dim2_ > out;
//
//    make_multi_iterator(0, 0).iterate([&](size_t a, size_t b) { out.push_back(dim2_{a, b}); });
//
//    ASSERT_EQ(0, out.size());
//}

template < typename T >
void print(array< T, 3 > a) {
    std::cout << a << std::endl;
}

TEST(test_hypercube_iterator, 3D) {
    hypercube_view< size_t, 3 > view{{0, 1, 0}, {2, 2, 2}};

    auto it = view.begin();
    array< size_t, 3 > tmp = it;

    //    for( int i = 0; i < 3; ++i )
    //    {
    //    it++;
    //    print(it);}

    for (auto it : view) {
        print(it);
    }
    //    std::cout << print(it) << std::endl;
}

TEST(test_hypercube_iterator, 3D_range) {
    auto view = make_hypercube_view(make_range(0, 2), make_range(1, 2), make_range(0, 2));

    auto it = view.begin();
    //    array< size_t, 3 > tmp = it;

    //    for( int i = 0; i < 3; ++i )
    //    {
    //    it++;
    //    print(it);}

    for (auto it : view) {
        print(it);
    }
    //    std::cout << print(it) << std::endl;
}

TEST(test_hypercube_iterator, 3D_brace_enclosed_init_list) {
    auto view = make_hypercube_view({0, 2}, {1, 2}, {0, 2});

    auto it = view.begin();
    //    array< size_t, 3 > tmp = it;

    //    for( int i = 0; i < 3; ++i )
    //    {
    //    it++;
    //    print(it);}

    for (auto it : view) {
        print(it);
    }
    //    std::cout << print(it) << std::endl;
}
