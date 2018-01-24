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

#include <common/defs.hpp>
#include <stencil-composition/icosahedral_grids/accessor.hpp>
#include <stencil-composition/icosahedral_grids/accessor_metafunctions.hpp>
#include <stencil-composition/icosahedral_grids/vector_accessor.hpp>
#include <stencil-composition/global_accessor.hpp>

TEST(accessor, is_accessor) {
    using namespace gridtools;
    GRIDTOOLS_STATIC_ASSERT(
        (is_accessor< accessor< 6, enumtype::inout, enumtype::cells, extent< 3, 4, 4, 5 > > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< accessor< 2, enumtype::in, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_accessor< int >::value), "");
}

TEST(accessor, is_accessor_readonly) {
    using namespace gridtools;
    GRIDTOOLS_STATIC_ASSERT((is_accessor_readonly< in_accessor< 0, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor_readonly< accessor< 0, enumtype::in, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor_readonly< vector_accessor< 0, enumtype::in, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor_readonly< global_accessor< 0 > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_accessor_readonly< inout_accessor< 0, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_accessor_readonly< accessor< 0, enumtype::inout, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT(
        (!is_accessor_readonly< vector_accessor< 0, enumtype::inout, enumtype::cells > >::value), "");
}

TEST(accessor, is_grid_accessor) {
    using namespace gridtools;
    GRIDTOOLS_STATIC_ASSERT((is_grid_accessor< accessor< 0, enumtype::in, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((is_grid_accessor< vector_accessor< 0, enumtype::in, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_grid_accessor< global_accessor< 0 > >::value), "");
}

TEST(accessor, is_regular_accessor) {
    using namespace gridtools;
    GRIDTOOLS_STATIC_ASSERT((is_regular_accessor< accessor< 0, enumtype::in, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_regular_accessor< vector_accessor< 0, enumtype::in, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_regular_accessor< global_accessor< 0 > >::value), "");
}

TEST(accessor, is_vector_accessor) {
    using namespace gridtools;
    GRIDTOOLS_STATIC_ASSERT((is_vector_accessor< vector_accessor< 0, enumtype::in, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_vector_accessor< accessor< 0, enumtype::in, enumtype::cells > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_vector_accessor< global_accessor< 0 > >::value), "");
}
