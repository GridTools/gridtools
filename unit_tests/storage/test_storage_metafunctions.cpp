/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include <storage/storage_metafunctions.hpp>
#include <storage/meta_storage.hpp>
#include <storage/storage.hpp>

using namespace gridtools;

TEST(storage_metafunctions, test_storage) {
    using namespace gridtools;
    using namespace enumtype;

    typedef gridtools::layout_map< 0, 1, 2 > layout_t;
    typedef meta_storage<
        meta_storage_aligned< meta_storage_base< static_int< 0 >, layout_t, false >, aligned< 32 >, halo< 0, 0, 0 > > >
        meta_data_t;
    typedef base_storage< wrap_pointer< double >, meta_data_t, 1 > base_st;
    typedef storage< base_st > storage_t;
    typedef no_storage_type_yet< storage_t > tmp_storage_t;
    // check metafunctions
    GRIDTOOLS_STATIC_ASSERT(is_any_storage< base_st >::value, "is_any_storage<base_storage<...>> failed");
    GRIDTOOLS_STATIC_ASSERT(is_any_storage< storage_t >::value, "is_any_storage<storage<...>> failed");
    GRIDTOOLS_STATIC_ASSERT(is_any_storage< tmp_storage_t >::value, "is_any_storage<tmp_storage_t<...>> failed");

    GRIDTOOLS_STATIC_ASSERT(
        is_actual_storage< pointer< base_st > >::value, "is_actual_storage<pointer<base_storage<...>>> failed");
    GRIDTOOLS_STATIC_ASSERT(
        is_actual_storage< pointer< storage_t > >::value, "is_actual_storage<pointer<storage_t<...>>> failed");
    GRIDTOOLS_STATIC_ASSERT(
        !is_actual_storage< pointer< tmp_storage_t > >::value, "is_actual_storage<pointer<tmp_storage_t<...>>> failed");

    GRIDTOOLS_STATIC_ASSERT(!is_temporary_storage< base_st >::value, "is_temporary_storage<base_storage<...>> failed");
    GRIDTOOLS_STATIC_ASSERT(!is_temporary_storage< storage_t >::value, "is_temporary_storage<storage_t<...>> failed");
    GRIDTOOLS_STATIC_ASSERT(
        is_temporary_storage< tmp_storage_t >::value, "is_temporary_storage<tmp_storage_t<...>> failed");
}
