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

template < typename T >
void print(array< T, 3 > a) {
    std::cout << a << std::endl;
}

TEST(test_hypercube_iterator, basic) {
    hypercube< size_t, 3 > cube3d(range< size_t >(1, 3), range< size_t >(5, 7), range< size_t >(1, 2));
    hypercube_view< size_t, 3 > view(cube3d);

    //    auto view = make_hypercube_view(make_range(0, 2), make_range(1, 2), make_range(0, 2));
    //
    for (auto it : view) {
        print(it);
    }
}

// TEST(test_hypercube_iterator, 3D_range) {
//    auto view = make_hypercube_view(make_range(0, 2), make_range(1, 2), make_range(0, 2));
//
//    for (auto it : view) {
//        print(it);
//    }
//}

// TEST(test_hypercube_iterator, 3D_brace_enclosed_init_list) {
//    auto view = make_hypercube_view({0, 2}, {1, 2}, {0, 2});
//
//    for (auto it : view) {
//        print(it);
//    }
//}
