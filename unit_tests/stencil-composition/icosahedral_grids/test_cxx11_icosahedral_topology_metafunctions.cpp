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
#include "common/defs.hpp"
#include "stencil-composition/icosahedral_grids/icosahedral_topology_metafunctions.hpp"

using namespace gridtools;

TEST(icosahedral_topology_metafunctions, selector_uuid) {
    // 0(cells) + 4+8+ /*(16)*/ + 32
    GRIDTOOLS_STATIC_ASSERT((impl::compute_uuid< enumtype::cells::value, selector< 1, 1, -1, 1 > >::value ==
                                enumtype::cells::value + 44 + enumtype::metastorage_library_indices_limit),
        "ERROR");

    // 0(cells) + 4+8+ /*(16)*/ + 32 //the rest of dimensions are ignored
    GRIDTOOLS_STATIC_ASSERT((impl::compute_uuid< enumtype::cells::value, selector< 1, 1, -1, 1, 1, 1 > >::value ==
                                enumtype::cells::value + 44 + enumtype::metastorage_library_indices_limit),
        "ERROR");

    // 0(cells) + 4+8+ /*(16)*/ + 32 //the rest of dimensions are ignored
    GRIDTOOLS_STATIC_ASSERT((impl::compute_uuid< enumtype::cells::value, selector< 1, 1, 1, 1, 1 > >::value ==
                                enumtype::cells::value + 60 + enumtype::metastorage_library_indices_limit),
        "ERROR");

    // 1(edges) + 4+/*8*/+ 16 + 32 //the rest of dimensions are ignored
    GRIDTOOLS_STATIC_ASSERT((impl::compute_uuid< enumtype::edges::value, selector< 1, -1, 1, 1, 1, 1 > >::value ==
                                enumtype::edges::value + 52 + enumtype::metastorage_library_indices_limit),
        "ERROR");
}

TEST(icosahedral_topology_metafunctions, array_dim_initializer) {

    constexpr auto array_ =
        impl::array_dim_initializers< uint_t, 4, location_type< 0, 2 >, selector< 1, 1, 1, 1 > >::apply(
            array< uint_t, 3 >{3, 4, 5});
    static_assert((array_.n_dimensions == 4), "error");
    static_assert((array_[0] == 3), "error");
    static_assert((array_[1] == 2), "error");
    static_assert((array_[2] == 4), "error");
    static_assert((array_[3] == 5), "error");

    constexpr auto array2_ =
        impl::array_dim_initializers< uint_t, 6, location_type< 0, 1 >, selector< 1, 1, -1, 1, 1, 1 > >::apply(
            array< uint_t, 3 >{3, 4, 5}, 7, 8);
    static_assert((array2_.n_dimensions == 6), "error");
    static_assert((array2_[0] == 3), "error");
    static_assert((array2_[1] == 1), "error");
    static_assert((array2_[3] == 5), "error");
    static_assert((array2_[4] == 7), "error");
    static_assert((array2_[5] == 8), "error");
}
