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
#include <gridtools/common/defs.hpp>
#include <gridtools/common/gt_assert.hpp>
#include <gridtools/stencil-composition/icosahedral_grids/icosahedral_topology_metafunctions.hpp>
#include <gridtools/stencil-composition/location_type.hpp>

using namespace gridtools;

TEST(icosahedral_topology_metafunctions, selector_uuid) {
    // 0(cells) + 4+8+ /*(16)*/ + 32
    GT_STATIC_ASSERT((impl::compute_uuid<enumtype::cells::value, selector<1, 1, 0, 1>>::value ==
                         enumtype::cells::value + 44 + metastorage_library_indices_limit),
        "ERROR");

    // 0(cells) + 4+8+ /*(16)*/ + 32 //the rest of dimensions are ignored
    GT_STATIC_ASSERT((impl::compute_uuid<enumtype::cells::value, selector<1, 1, 0, 1, 1, 1>>::value ==
                         enumtype::cells::value + 44 + metastorage_library_indices_limit),
        "ERROR");

    // 0(cells) + 4+8+ /*(16)*/ + 32 //the rest of dimensions are ignored
    GT_STATIC_ASSERT((impl::compute_uuid<enumtype::cells::value, selector<1, 1, 1, 1, 1>>::value ==
                         enumtype::cells::value + 60 + metastorage_library_indices_limit),
        "ERROR");

    // 1(edges) + 4+/*8*/+ 16 + 32 //the rest of dimensions are ignored
    GT_STATIC_ASSERT((impl::compute_uuid<enumtype::edges::value, selector<1, 0, 1, 1, 1, 1>>::value ==
                         enumtype::edges::value + 52 + metastorage_library_indices_limit),
        "ERROR");
}
