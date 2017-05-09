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
#include <storage/storage-facility.hpp>

using namespace gridtools;

TEST(storage_info, test_extender) {

    typedef gridtools::layout_map< 0, 1, 2, 3, 4 > layout_t;
    typedef typename gridtools::meta_storage_base< static_int< 0 >, layout_t, false > meta_t;
    typedef typename gridtools::meta_storage_aligned< meta_t, aligned< 1 >, halo< 0, 0, 0, 0, 0 > > aligned_meta_t;

    aligned_meta_t meta_{11u, 12u, 13u, 14u, 15u};

    meta_storage_extender meta_extended_;
    // NOTE: the extended meta_storage in not a literal type
    auto m = meta_extended_(meta_, 5);
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< decltype(m),
            meta_storage_aligned< meta_storage_base< static_int< 0 >, layout_map< 1, 2, 3, 4, 5, 0 >, false >,
                             aligned< 1u >,
                             halo< 0u, 0u, 0u, 0u, 0u, 0u > > >::value),
        "Error");
}
