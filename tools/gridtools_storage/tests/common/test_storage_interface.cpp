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

#include "common/storage_interface.hpp"

using namespace gridtools;

// storage that implements the storage_interface
struct host_storage : storage_interface<host_storage> {
    int a;
    host_storage(unsigned i) : a(i) {}
    void clone_to_device_impl() { a*=2; }
};

// another storage that implements the storage_interface
struct cuda_storage : storage_interface<cuda_storage> {
    int a;
    cuda_storage(unsigned i) : a(i) {}
    void clone_to_device_impl() { a*=3; }
};

TEST(StorageInterface, Simple) {
    host_storage h(10);
    cuda_storage c(10);

    EXPECT_EQ(h.a, 10);
    EXPECT_EQ(c.a, 10);

    h.clone_to_device();
    c.clone_to_device();
    
    EXPECT_EQ(h.a, 20);
    EXPECT_EQ(c.a, 30);
}

TEST(StorageInterface, Sizes) {
    cuda_storage c(10);
    host_storage h(10);
    // the sizes should stay 1, because the idea is that the 
    // storage interface is only providing a set of methods
    EXPECT_EQ(sizeof(storage_interface<host_storage>), 1);
    EXPECT_EQ(sizeof(storage_interface<cuda_storage>), 1);
}
