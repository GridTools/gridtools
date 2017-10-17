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
#include <storage/common/storage_info_interface.hpp>

using namespace gridtools;

TEST(StorageInfo, Simple) {
    {
        storage_info_interface< 0, layout_map< 2, 1, 0 > > si(3, 3, 3);
        EXPECT_EQ((si.index(0, 0, 0)), 0);
        EXPECT_EQ((si.index(0, 0, 1)), 9);
        EXPECT_EQ((si.index(0, 0, 2)), 18);

        EXPECT_EQ((si.index(0, 1, 0)), 3);
        EXPECT_EQ((si.index(0, 1, 1)), 12);
        EXPECT_EQ((si.index(0, 1, 2)), 21);

        EXPECT_EQ((si.index(0, 2, 0)), 6);
        EXPECT_EQ((si.index(0, 2, 1)), 15);
        EXPECT_EQ((si.index(0, 2, 2)), 24);

        EXPECT_EQ((si.index(1, 0, 0)), 1);
        EXPECT_EQ((si.index(1, 0, 1)), 10);
        EXPECT_EQ((si.index(1, 0, 2)), 19);
    }
    {
        storage_info_interface< 0, layout_map< 0, 1, 2 > > si(3, 3, 3);
        EXPECT_EQ((si.index(0, 0, 0)), 0);
        EXPECT_EQ((si.index(0, 0, 1)), 1);
        EXPECT_EQ((si.index(0, 0, 2)), 2);

        EXPECT_EQ((si.index(0, 1, 0)), 3);
        EXPECT_EQ((si.index(0, 1, 1)), 4);
        EXPECT_EQ((si.index(0, 1, 2)), 5);

        EXPECT_EQ((si.index(0, 2, 0)), 6);
        EXPECT_EQ((si.index(0, 2, 1)), 7);
        EXPECT_EQ((si.index(0, 2, 2)), 8);

        EXPECT_EQ((si.index(1, 0, 0)), 9);
        EXPECT_EQ((si.index(1, 0, 1)), 10);
        EXPECT_EQ((si.index(1, 0, 2)), 11);
    }
    {
        storage_info_interface< 0, layout_map< 1, 0, 2 > > si(3, 3, 3);
        EXPECT_EQ((si.index(0, 0, 0)), 0);
        EXPECT_EQ((si.index(0, 0, 1)), 1);
        EXPECT_EQ((si.index(0, 0, 2)), 2);

        EXPECT_EQ((si.index(0, 1, 0)), 9);
        EXPECT_EQ((si.index(0, 1, 1)), 10);
        EXPECT_EQ((si.index(0, 1, 2)), 11);

        EXPECT_EQ((si.index(0, 2, 0)), 18);
        EXPECT_EQ((si.index(0, 2, 1)), 19);
        EXPECT_EQ((si.index(0, 2, 2)), 20);

        EXPECT_EQ((si.index(1, 0, 0)), 3);
        EXPECT_EQ((si.index(1, 0, 1)), 4);
        EXPECT_EQ((si.index(1, 0, 2)), 5);
    }

    // storage info has to be constexpr capable
    constexpr storage_info_interface< 0, layout_map< 1, 0, 2 > > si(3, 3, 3);
    GRIDTOOLS_STATIC_ASSERT(si.padded_total_length() == 27, "storage info is not constexpr anymore");
    GRIDTOOLS_STATIC_ASSERT(si.total_length() == 27, "storage info is not constexpr anymore");
    GRIDTOOLS_STATIC_ASSERT(si.length() == 27, "storage info is not constexpr anymore");
    GRIDTOOLS_STATIC_ASSERT(si.stride< 0 >() == 3, "storage info is not constexpr anymore");
    GRIDTOOLS_STATIC_ASSERT(si.stride< 1 >() == 9, "storage info is not constexpr anymore");
    GRIDTOOLS_STATIC_ASSERT(si.stride< 2 >() == 1, "storage info is not constexpr anymore");
    GRIDTOOLS_STATIC_ASSERT(si.index(0, 1, 0) == 9, "storage info is not constexpr anymore");
    GRIDTOOLS_STATIC_ASSERT(si.index(1, 0, 0) == 3, "storage info is not constexpr anymore");
    GRIDTOOLS_STATIC_ASSERT(si.index(0, 0, 1) == 1, "storage info is not constexpr anymore");

    // test wiht different dims
    storage_info_interface< 0, layout_map< 1, 2, 3, 0 > > x(5, 7, 8, 2);
    EXPECT_EQ((x.dim< 0 >()), 5);
    EXPECT_EQ((x.dim< 1 >()), 7);
    EXPECT_EQ((x.dim< 2 >()), 8);
    EXPECT_EQ((x.dim< 3 >()), 2);

    EXPECT_EQ((x.stride< 0 >()), 56);
    EXPECT_EQ((x.stride< 1 >()), 8);
    EXPECT_EQ((x.stride< 2 >()), 1);
    EXPECT_EQ((x.stride< 3 >()), 280);
}

TEST(StorageInfo, ArrayAccess) {
    {
        storage_info_interface< 0, layout_map< 2, 1, 0 > > si(3, 3, 3);
        EXPECT_EQ((si.index({0, 0, 0})), 0);
        EXPECT_EQ((si.index({0, 0, 1})), 9);
        EXPECT_EQ((si.index({0, 0, 2})), 18);

        EXPECT_EQ((si.index({0, 1, 0})), 3);
        EXPECT_EQ((si.index({0, 1, 1})), 12);
        EXPECT_EQ((si.index({0, 1, 2})), 21);

        EXPECT_EQ((si.index({0, 2, 0})), 6);
        EXPECT_EQ((si.index({0, 2, 1})), 15);
        EXPECT_EQ((si.index({0, 2, 2})), 24);

        EXPECT_EQ((si.index({1, 0, 0})), 1);
        EXPECT_EQ((si.index({1, 0, 1})), 10);
        EXPECT_EQ((si.index({1, 0, 2})), 19);
    }
    {
        storage_info_interface< 0, layout_map< 0, 1, 2 > > si(3, 3, 3);
        EXPECT_EQ((si.index({0, 0, 0})), 0);
        EXPECT_EQ((si.index({0, 0, 1})), 1);
        EXPECT_EQ((si.index({0, 0, 2})), 2);

        EXPECT_EQ((si.index({0, 1, 0})), 3);
        EXPECT_EQ((si.index({0, 1, 1})), 4);
        EXPECT_EQ((si.index({0, 1, 2})), 5);

        EXPECT_EQ((si.index({0, 2, 0})), 6);
        EXPECT_EQ((si.index({0, 2, 1})), 7);
        EXPECT_EQ((si.index({0, 2, 2})), 8);

        EXPECT_EQ((si.index({1, 0, 0})), 9);
        EXPECT_EQ((si.index({1, 0, 1})), 10);
        EXPECT_EQ((si.index({1, 0, 2})), 11);
    }
    {
        storage_info_interface< 0, layout_map< 1, 0, 2 > > si(3, 3, 3);
        EXPECT_EQ((si.index({0, 0, 0})), 0);
        EXPECT_EQ((si.index({0, 0, 1})), 1);
        EXPECT_EQ((si.index({0, 0, 2})), 2);

        EXPECT_EQ((si.index({0, 1, 0})), 9);
        EXPECT_EQ((si.index({0, 1, 1})), 10);
        EXPECT_EQ((si.index({0, 1, 2})), 11);

        EXPECT_EQ((si.index({0, 2, 0})), 18);
        EXPECT_EQ((si.index({0, 2, 1})), 19);
        EXPECT_EQ((si.index({0, 2, 2})), 20);

        EXPECT_EQ((si.index({1, 0, 0})), 3);
        EXPECT_EQ((si.index({1, 0, 1})), 4);
        EXPECT_EQ((si.index({1, 0, 2})), 5);
    }
}

TEST(StorageInfo, Halo) {
    // test with simple halo, dims and strides are extended
    storage_info_interface< 0, layout_map< 2, 1, 0 >, halo< 2, 2, 2 > > x(7, 7, 7);
    EXPECT_EQ((x.dim< 0 >()), 7);
    EXPECT_EQ((x.dim< 1 >()), 7);
    EXPECT_EQ((x.dim< 2 >()), 7);

    EXPECT_EQ((x.stride< 0 >()), 1);
    EXPECT_EQ((x.stride< 1 >()), 7);
    EXPECT_EQ((x.stride< 2 >()), 49);

    // test with simple halo, dims and strides are extended
    storage_info_interface< 0, layout_map< 0, 1, 2 >, halo< 2, 2, 2 > > y(7, 7, 7);
    EXPECT_EQ((y.dim< 0 >()), 7);
    EXPECT_EQ((y.dim< 1 >()), 7);
    EXPECT_EQ((y.dim< 2 >()), 7);

    EXPECT_EQ((y.stride< 0 >()), 49);
    EXPECT_EQ((y.stride< 1 >()), 7);
    EXPECT_EQ((y.stride< 2 >()), 1);

    // test with heterogeneous halo, dims and strides are extended
    storage_info_interface< 0, layout_map< 2, 1, 0 >, halo< 2, 4, 0 > > z(7, 11, 3);
    EXPECT_EQ((z.dim< 0 >()), 7);
    EXPECT_EQ((z.dim< 1 >()), 11);
    EXPECT_EQ((z.dim< 2 >()), 3);

    EXPECT_EQ((z.stride< 0 >()), 1);
    EXPECT_EQ((z.stride< 1 >()), 7);
    EXPECT_EQ((z.stride< 2 >()), 77);
}

TEST(StorageInfo, Alignment) {
    {
        // test with different dims and alignment
        storage_info_interface< 0, layout_map< 1, 2, 3, 0 >, halo< 0, 0, 0, 0 >, alignment< 32 > > x(5, 7, 32, 2);
        EXPECT_EQ((x.dim< 0 >()), 5);
        EXPECT_EQ((x.dim< 1 >()), 7);
        EXPECT_EQ((x.dim< 2 >()), 32);
        EXPECT_EQ((x.dim< 3 >()), 2);

        EXPECT_EQ((x.stride< 0 >()), 32 * 7);
        EXPECT_EQ((x.stride< 1 >()), 32);
        EXPECT_EQ((x.stride< 2 >()), 1);
        EXPECT_EQ((x.stride< 3 >()), 5 * 32 * 7);
    }
    {
        // test with different dims, halo and alignment
        storage_info_interface< 0, layout_map< 1, 2, 3, 0 >, halo< 1, 2, 3, 4 >, alignment< 32 > > x(7, 11, 32, 10);
        EXPECT_EQ((x.dim< 0 >()), 7);
        EXPECT_EQ((x.dim< 1 >()), 11);
        EXPECT_EQ((x.dim< 2 >()), 32);
        EXPECT_EQ((x.dim< 3 >()), 10);

        EXPECT_EQ((x.stride< 0 >()), 32 * 11);
        EXPECT_EQ((x.stride< 1 >()), 32);
        EXPECT_EQ((x.stride< 2 >()), 1);
        EXPECT_EQ((x.stride< 3 >()), 32 * 11 * 7);

        EXPECT_EQ(x.index(0, 0, 0, 0), 29); // halo point
        EXPECT_EQ(x.index(0, 0, 1, 0), 30); // halo point
        EXPECT_EQ(x.index(0, 0, 2, 0), 31); // halo point
        EXPECT_EQ(x.index(0, 0, 3, 0), 32); // first data point, aligned
    }
    {
        // test with different dims, halo and alignment
        storage_info_interface< 0, layout_map< 3, 2, 1, 0 >, halo< 1, 2, 3, 4 >, alignment< 32 > > x(32, 11, 14, 10);
        EXPECT_EQ((x.dim< 0 >()), 32);
        EXPECT_EQ((x.dim< 1 >()), 11);
        EXPECT_EQ((x.dim< 2 >()), 14);
        EXPECT_EQ((x.dim< 3 >()), 10);

        EXPECT_EQ((x.stride< 0 >()), 1);
        EXPECT_EQ((x.stride< 1 >()), 32);
        EXPECT_EQ((x.stride< 2 >()), 32 * 11);
        EXPECT_EQ((x.stride< 3 >()), 32 * 11 * 14);

        EXPECT_EQ(x.index(0, 0, 0, 0), 31); // halo point
        EXPECT_EQ(x.index(0, 1, 0, 0), 31 + 32);
        EXPECT_EQ(x.index(0, 0, 1, 0), 31 + 32 * 11);
        EXPECT_EQ(x.index(0, 0, 0, 1), 31 + 32 * 11 * 14);
    }
    {
        // test with masked dimensions
        storage_info_interface< 0, layout_map< 1, -1, -1, 0 >, halo< 1, 2, 3, 4 >, alignment< 32 > > x(7, 7, 8, 10);
        EXPECT_EQ((x.dim< 0 >()), 7);
        EXPECT_EQ((x.dim< 1 >()), 1);
        EXPECT_EQ((x.dim< 2 >()), 1);
        EXPECT_EQ((x.dim< 3 >()), 10);

        EXPECT_EQ((x.stride< 0 >()), 1);
        EXPECT_EQ((x.stride< 1 >()), 0);
        EXPECT_EQ((x.stride< 2 >()), 0);
        EXPECT_EQ((x.stride< 3 >()), 7);

        EXPECT_EQ(x.index(0, 0, 0, 0), 31); // halo point
        EXPECT_EQ(x.index(0, 1, 0, 0), 31);
        EXPECT_EQ(x.index(0, 0, 1, 0), 31);
        EXPECT_EQ(x.index(0, 0, 0, 1), 31 + 7);

        EXPECT_EQ(x.padded_total_length(), 7 * 10 + 31);
        EXPECT_EQ(x.total_length(), 7 * 10);
        EXPECT_EQ(x.length(), 5 * 2);
    }
}

TEST(StorageInfo, BeginEnd) {
    // no halo, no alignment
    storage_info_interface< 0, layout_map< 1, 2, 0 > > x(7, 7, 7);
    EXPECT_EQ(x.length(), 7 * 7 * 7);
    EXPECT_EQ(x.total_length(), 7 * 7 * 7);
    EXPECT_EQ(x.padded_total_length(), 7 * 7 * 7);
    EXPECT_EQ(x.begin(), 0);
    EXPECT_EQ(x.end(), 7 * 7 * 7 - 1);
    EXPECT_EQ(x.total_begin(), 0);
    EXPECT_EQ(x.total_end(), 7 * 7 * 7 - 1);

    // halo, no alignment
    storage_info_interface< 0, layout_map< 1, 2, 0 >, halo< 1, 2, 3 > > y(9, 11, 13);
    EXPECT_EQ(y.length(), 7 * 7 * 7);
    EXPECT_EQ(y.total_length(), 9 * 11 * 13);
    EXPECT_EQ(y.padded_total_length(), 9 * 11 * 13);
    EXPECT_EQ(y.begin(), y.index(1, 2, 3));
    EXPECT_EQ(y.end(), y.index(7, 8, 9));
    EXPECT_EQ(y.total_begin(), y.index(0, 0, 0));
    EXPECT_EQ(y.total_end(), y.index(8, 10, 12));

    // halo, alignment
    storage_info_interface< 0, layout_map< 1, 2, 0 >, halo< 1, 2, 3 >, alignment< 16 > > z(9, 11, 13);
    EXPECT_EQ(z.length(), 7 * 7 * 7);
    EXPECT_EQ(z.total_length(), 9 * 11 * 13);
    EXPECT_EQ(z.padded_total_length(), 9 * 16 * 13 + z.get_initial_offset());
    EXPECT_EQ(z.begin(), z.index(1, 2, 3));
    EXPECT_EQ(z.end(), z.index(7, 8, 9));
    EXPECT_EQ(z.total_begin(), z.index(0, 0, 0));
    EXPECT_EQ(z.total_end(), z.index(8, 10, 12));
}
