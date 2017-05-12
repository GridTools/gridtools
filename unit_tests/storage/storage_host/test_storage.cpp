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

#include "storage/storage_host/storage.hpp"

TEST(StorageHostTest, Simple) {
    // create two storages
    gridtools::host_storage< int > s1(2);
    gridtools::host_storage< int > s2(2);
    // test the is_storage check
    static_assert(gridtools::is_storage< decltype(s1) >::type::value, "is_storage check is not working anymore");
    static_assert(!gridtools::is_storage< int >::type::value, "is_storage check is not working anymore");
    // write some values
    s1.get_cpu_ptr()[0] = 10;
    s1.get_cpu_ptr()[1] = 20;
    s2.get_cpu_ptr()[0] = 100;
    s2.get_cpu_ptr()[1] = 200;
    // check if they are there
    EXPECT_EQ(s1.get_cpu_ptr()[1], 20);
    EXPECT_EQ(s1.get_cpu_ptr()[0], 10);
    EXPECT_EQ(s2.get_cpu_ptr()[1], 200);
    EXPECT_EQ(s2.get_cpu_ptr()[0], 100);
    // ptr ref should be equal to the cpu ptr
    EXPECT_EQ(s1.get_cpu_ptr(), s1.get_ptrs< typename gridtools::host_storage< int >::ptrs_t >());
    EXPECT_EQ(s2.get_cpu_ptr(), s2.get_ptrs< typename gridtools::host_storage< int >::ptrs_t >());
    // manually exchange the ptrs
    auto tmp = s1.get_cpu_ptr();
    s1.set_ptrs(s2.get_cpu_ptr());
    s2.set_ptrs(tmp);
    // check if changes are there
    EXPECT_EQ(s2.get_cpu_ptr()[1], 20);
    EXPECT_EQ(s2.get_cpu_ptr()[0], 10);
    EXPECT_EQ(s1.get_cpu_ptr()[1], 200);
    EXPECT_EQ(s1.get_cpu_ptr()[0], 100);
}

TEST(StorageHostTest, InitializedStorage) {
    // create two storages
    gridtools::host_storage< int > s1(2, 3);
    gridtools::host_storage< int > s2(2, 5);
    // check values
    EXPECT_EQ(s1.get_cpu_ptr()[0], 3);
    EXPECT_EQ(s1.get_cpu_ptr()[1], 3);
    EXPECT_EQ(s2.get_cpu_ptr()[0], 5);
    EXPECT_EQ(s2.get_cpu_ptr()[1], 5);
    // change one value
    s1.get_cpu_ptr()[0] = 10;
    s2.get_cpu_ptr()[1] = 20;
    // check values
    EXPECT_EQ(s1.get_cpu_ptr()[0], 10);
    EXPECT_EQ(s1.get_cpu_ptr()[1], 3);
    EXPECT_EQ(s2.get_cpu_ptr()[0], 5);
    EXPECT_EQ(s2.get_cpu_ptr()[1], 20);
}
