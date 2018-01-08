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

#include <common/array.hpp>
#include <gtest/gtest.h>
#include "interface/multi_iterator.hpp"

using namespace gridtools;

TEST(multi_iterator, 3D) {
    gridtools::array< uint_t, 3 > dims{2, 4, 3};
    using dim3 = std::tuple< int, int, int >;

    std::vector< dim3 > out;
    iterate(dims, [&](int a, int b, int c) { out.push_back(dim3(a, b, c)); });

    for (size_t i = 0; i < dims[0]; ++i)
        for (size_t j = 0; j < dims[1]; ++j)
            for (size_t k = 0; k < dims[2]; ++k)
                ASSERT_EQ(1, std::count(out.begin(), out.end(), dim3(i, j, k)));
}

TEST(multi_iterator, 2D) {
    gridtools::array< uint_t, 2 > dims{2, 43};
    using dim2 = std::tuple< int, int >;

    std::vector< dim2 > out;
    iterate(dims, [&](int a, int b) { out.push_back(dim2(a, b)); });

    for (size_t i = 0; i < dims[0]; ++i)
        for (size_t j = 0; j < dims[1]; ++j)
            ASSERT_EQ(1, std::count(out.begin(), out.end(), dim2(i, j)));
}

TEST(multi_iterator, 0D) {
    gridtools::array< uint_t, 0 > dims;
    std::vector< int > out;
    iterate(dims, [&]() { out.push_back(0); });

    ASSERT_EQ(0, out.size());
}

TEST(multi_iterator, 2D_with_size_zero) {
    gridtools::array< uint_t, 2 > dims{0, 0};
    using dim2 = std::tuple< int, int >;

    std::vector< dim2 > out;
    iterate(dims, [&](int a, int b) { out.push_back(dim2(a, b)); });

    ASSERT_EQ(0, out.size());
}
