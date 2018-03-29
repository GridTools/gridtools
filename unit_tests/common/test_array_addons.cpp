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
#include "common/array_addons.hpp"
#include "common/defs.hpp"
#include "common/array.hpp"
#include "common/tuple.hpp"
#include "common/pair.hpp"
#include "gtest/gtest.h"
#include <cstddef>

using namespace gridtools;

TEST(array, to_vector) {
    array< uint_t, 4 > a{1, 2, 3, 4};

    auto v = to_vector(a);

    ASSERT_EQ(4, v.size());

    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(a[i], v[i]);
    }
}

TEST(transpose, array_3x2) {
    gridtools::array< gridtools::array< size_t, 2 >, 3 > in{{{11, 12}, {21, 22}, {31, 32}}};
    gridtools::array< gridtools::array< size_t, 3 >, 2 > ref{{{11, 21, 31}, {12, 22, 32}}};

    auto result = transpose(in);

    ASSERT_EQ(ref, result);
}

TEST(transpose, array_3x1) {
    gridtools::array< gridtools::array< size_t, 2 >, 3 > in{{{1}, {2}, {3}}};
    gridtools::array< gridtools::array< size_t, 3 >, 2 > ref{{{1, 2, 3}}};

    auto result = transpose(in);

    ASSERT_EQ(ref, result);
}

TEST(transpose, array_of_pairs) {
    gridtools::array< gridtools::pair< size_t, size_t >, 3 > in{{gridtools::pair< size_t, size_t >{11, 12},
        gridtools::pair< size_t, size_t >{21, 22},
        gridtools::pair< size_t, size_t >{31, 32}}};
    gridtools::array< gridtools::array< size_t, 3 >, 2 > ref{{{11, 21, 31}, {12, 22, 32}}};

    auto result = transpose(in);

    ASSERT_EQ(ref, result);
}

TEST(transpose, pair_of_pairs) {
    using pair_of_pairs = gridtools::pair< gridtools::pair< size_t, size_t >, gridtools::pair< size_t, size_t > >;
    pair_of_pairs in{gridtools::pair< size_t, size_t >{11, 12}, gridtools::pair< size_t, size_t >{21, 22}};
    gridtools::array< gridtools::array< size_t, 2 >, 2 > ref{{{11, 21}, {12, 22}}};

    auto result = transpose(in);

    ASSERT_EQ(ref, result);
}

// enable once gridtools::tuple is more std-compliant
// TEST(tuple, transpose) {
//    gridtools::array< gridtools::tuple< size_t, size_t >, 3 > in{{{11, 12}, {21, 22}, {31, 32}}};
//    gridtools::array< gridtools::array< size_t, 3 >, 2 > ref{{{11, 21, 31}, {12, 22, 32}}};
//    auto result = transpose(in);
//
//    ASSERT_EQ(ref, result);
//}

TEST(convert_to, size_t_to_int) {
    const size_t val0 = 1;
    const size_t val1 = 2;
    gridtools::array< size_t, 2 > in{val0, val1};

    auto result = convert_to< int >(in);

    ASSERT_TYPE_EQ< gridtools::array< int, 2 >, decltype(result) >();
    ASSERT_EQ((int)val0, result[0]);
    ASSERT_EQ((int)val1, result[1]);
}

TEST(convert_to, size_t_to_double) {
    const size_t val0 = 1;
    const size_t val1 = 2;
    gridtools::array< size_t, 2 > in{val0, val1};

    auto result = convert_to< double >(in);

    ASSERT_TYPE_EQ< gridtools::array< double, 2 >, decltype(result) >();
    ASSERT_EQ((double)val0, result[0]);
    ASSERT_EQ((double)val1, result[1]);
}

TEST(convert_to, from_pair) {
    const int val0 = 1;
    const int val1 = 2;
    gridtools::pair< size_t, size_t > in{val0, val1};

    auto result = convert_to< double >(in);

    ASSERT_TYPE_EQ< gridtools::array< double, 2 >, decltype(result) >();
    ASSERT_EQ((double)val0, result[0]);
    ASSERT_EQ((double)val1, result[1]);
}
