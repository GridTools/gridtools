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

#include <gridtools/common/gt_assert.hpp>
#include <gridtools/storage/storage-facility.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <iostream>
#include <utility>

namespace gt = gridtools;

TEST(Storage, Swap) {
    using storage_info_t = gt::storage_traits<backend_t::backend_id_t>::storage_info_t<0, 3>;
    using data_store_t = gt::storage_traits<backend_t::backend_id_t>::data_store_t<double, storage_info_t>;

    storage_info_t s1(3, 3, 3);
    data_store_t ds1(s1, "ds1");

    storage_info_t s2(3, 30, 20);
    data_store_t ds2(s2, "ds2");

    auto name1 = ds1.name();
    auto ptr1 = ds1.get_storage_ptr();
    auto iptr1 = ds1.get_storage_info_ptr();

    auto name2 = ds2.name();
    auto ptr2 = ds2.get_storage_ptr();
    auto iptr2 = ds2.get_storage_info_ptr();

    using std::swap;
    swap(ds1, ds2);

    EXPECT_EQ(name1, ds2.name());
    EXPECT_EQ(ptr1, ds2.get_storage_ptr());
    EXPECT_EQ(iptr1, ds2.get_storage_info_ptr());

    EXPECT_EQ(name2, ds1.name());
    EXPECT_EQ(ptr2, ds1.get_storage_ptr());
    EXPECT_EQ(iptr2, ds1.get_storage_info_ptr());
}
