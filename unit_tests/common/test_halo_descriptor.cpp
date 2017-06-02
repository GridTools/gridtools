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
#include "common/halo_descriptor.hpp"

using namespace gridtools;

TEST(test_halo_descriptor, empty_compute_domain) {
    uint_t size = 6;
    uint_t halo_size = 3;

    ASSERT_ANY_THROW((halo_descriptor{halo_size, halo_size, halo_size, size - halo_size - 1, size}));
}

TEST(test_halo_descriptor, begin_in_halo) {
    uint_t begin = 0;
    uint_t halo_size = 1;
    uint_t size = 10;

    ASSERT_ANY_THROW((halo_descriptor{halo_size, halo_size, begin, size - halo_size - 1, size}));
}

TEST(test_halo_descriptor, end_in_halo) {
    uint_t halo_size = 1;
    uint_t size = 10;
    uint_t end = size - 1;

    ASSERT_ANY_THROW((halo_descriptor{halo_size, halo_size, halo_size, end, size}));
}

TEST(test_halo_descriptor, invalid_total_length) {
    uint_t halo_size = 3;
    uint_t begin = halo_size;
    uint_t end = 10 - halo_size - 1;
    uint_t size = 9;

    ASSERT_ANY_THROW((halo_descriptor{halo_size, halo_size, begin, end, size}));
}

TEST(test_halo_descriptor, is_valid) {
    uint_t size = 7;
    uint_t halo_size = 3;

    ASSERT_NO_THROW((halo_descriptor{halo_size, halo_size, halo_size, size - halo_size - 1, size}));
}

TEST(test_halo_descriptor, default_constructed_is_valid) { ASSERT_NO_THROW((halo_descriptor())); }
