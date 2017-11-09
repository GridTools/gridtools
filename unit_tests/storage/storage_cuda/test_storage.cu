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

#include <common/gt_assert.hpp>
#include <storage/storage_cuda/cuda_storage.hpp>

__global__ void initial_check_s1(int *s) {
    ASSERT_OR_THROW((s[0] == 10), "check failed");
    ASSERT_OR_THROW((s[1] == (s[1] == 10)), "check failed");
    if (s[0] != 10 || s[1] != 10) {
        s[0] = -1;
        s[1] = -1;
    }
}

__global__ void check_s1(int *s) {
    ASSERT_OR_THROW((s[0] == 10), "check failed");
    ASSERT_OR_THROW((s[1] == 20), "check failed");
    s[0] = 30;
    s[1] = 40;
}

__global__ void check_s2(int *s) {
    ASSERT_OR_THROW((s[0] == 100), "check failed");
    ASSERT_OR_THROW((s[1] == 200), "check failed");
    s[0] = 300;
    s[1] = 400;
}

TEST(StorageHostTest, Simple) {
    // create two storages
    gridtools::cuda_storage< int > s1(2);
    gridtools::cuda_storage< int > s2(2);
    // test the is_storage check
    GRIDTOOLS_STATIC_ASSERT(
        gridtools::is_storage< decltype(s1) >::type::value, "is_storage check is not working anymore");
    GRIDTOOLS_STATIC_ASSERT(!gridtools::is_storage< int >::type::value, "is_storage check is not working anymore");
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

    // clone to device
    s1.clone_to_device();
    s2.clone_to_device();
    // assert if the values were not copied correctly and reset values
    check_s1<<< 1, 1 >>>(s1.get_gpu_ptr());
    check_s2<<< 1, 1 >>>(s2.get_gpu_ptr());
    // clone_back
    s1.clone_from_device();
    s2.clone_from_device();
    // check values
    EXPECT_EQ(s1.get_cpu_ptr()[1], 40);
    EXPECT_EQ(s1.get_cpu_ptr()[0], 30);
    EXPECT_EQ(s2.get_cpu_ptr()[1], 400);
    EXPECT_EQ(s2.get_cpu_ptr()[0], 300);

    // ptr ref should be equal to the cpu ptr
    EXPECT_EQ(s1.get_cpu_ptr(), s1.get_ptrs< gridtools::cuda_storage< int >::ptrs_t >()[0]);
    EXPECT_EQ(s2.get_cpu_ptr(), s2.get_ptrs< gridtools::cuda_storage< int >::ptrs_t >()[0]);
    EXPECT_EQ(s1.get_gpu_ptr(), s1.get_ptrs< gridtools::cuda_storage< int >::ptrs_t >()[1]);
    EXPECT_EQ(s2.get_gpu_ptr(), s2.get_ptrs< gridtools::cuda_storage< int >::ptrs_t >()[1]);
    // swap the storages
    s1.swap(s2);
    // check if changes are there
    EXPECT_EQ(s2.get_cpu_ptr()[1], 40);
    EXPECT_EQ(s2.get_cpu_ptr()[0], 30);
    EXPECT_EQ(s1.get_cpu_ptr()[1], 400);
    EXPECT_EQ(s1.get_cpu_ptr()[0], 300);
    EXPECT_EQ(s2.get_gpu_ptr(), tmp[1]);
    EXPECT_EQ(s2.get_cpu_ptr(), tmp[0]);
}

TEST(StorageHostTest, InitializedStorage) {
    // create two storages
    gridtools::cuda_storage< int > s1(2, 10);
    // initial check
    initial_check_s1<<< 1, 1 >>>(s1.get_gpu_ptr());
    s1.clone_from_device();
    // check values
    EXPECT_EQ(s1.get_cpu_ptr()[0], 10);
    EXPECT_EQ(s1.get_cpu_ptr()[1], 10);
    // change one value
    s1.get_cpu_ptr()[1] = 20;
    // check values
    EXPECT_EQ(s1.get_cpu_ptr()[0], 10);
    EXPECT_EQ(s1.get_cpu_ptr()[1], 20);
    // some device things
    s1.clone_to_device();
    check_s1<<< 1, 1 >>>(s1.get_gpu_ptr());
    s1.clone_from_device();
    // check again
    EXPECT_EQ(s1.get_cpu_ptr()[0], 30);
    EXPECT_EQ(s1.get_cpu_ptr()[1], 40);
}
