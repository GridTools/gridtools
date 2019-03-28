/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/storage/common/storage_info_rt.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;

TEST(StorageInfoRT, Make3D) {
    using storage_info_t = storage_traits<backend_t>::storage_info_t<0, 3>;
    storage_info_t si(4, 5, 6);

    auto storage_info_rt_ = make_storage_info_rt(si);

    auto dims = storage_info_rt_.total_lengths();
    ASSERT_EQ(si.total_length<0>(), dims[0]);
    ASSERT_EQ(si.total_length<1>(), dims[1]);
    ASSERT_EQ(si.total_length<2>(), dims[2]);

    auto padded_lengths = storage_info_rt_.padded_lengths();
    ASSERT_EQ(si.padded_length<0>(), padded_lengths[0]);
    ASSERT_EQ(si.padded_length<1>(), padded_lengths[1]);
    ASSERT_EQ(si.padded_length<2>(), padded_lengths[2]);

    auto strides = storage_info_rt_.strides();
    ASSERT_EQ(si.stride<0>(), strides[0]);
    ASSERT_EQ(si.stride<1>(), strides[1]);
    ASSERT_EQ(si.stride<2>(), strides[2]);
}

TEST(StorageInfoRT, Make3Dmasked) {
    using storage_info_t = storage_traits<backend_t>::special_storage_info_t<0, selector<1, 0, 1>>;
    storage_info_t si(4, 5, 6);

    auto storage_info_rt_ = make_storage_info_rt(si);

    auto total_lengths = storage_info_rt_.total_lengths();
    ASSERT_EQ(si.total_length<0>(), total_lengths[0]);
    ASSERT_EQ(si.total_length<1>(), total_lengths[1]);
    ASSERT_EQ(si.total_length<2>(), total_lengths[2]);

    auto padded_lengths = storage_info_rt_.padded_lengths();
    ASSERT_EQ(si.padded_length<0>(), padded_lengths[0]);
    ASSERT_EQ(si.padded_length<1>(), padded_lengths[1]);
    ASSERT_EQ(si.padded_length<2>(), padded_lengths[2]);

    auto strides = storage_info_rt_.strides();
    ASSERT_EQ(si.stride<0>(), strides[0]);
    ASSERT_EQ(si.stride<1>(), strides[1]);
    ASSERT_EQ(si.stride<2>(), strides[2]);
}
