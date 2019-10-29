/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gridtools/storage/common/storage_info_rt.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;
using testing::ElementsAreArray;

TEST(StorageInfoRT, Make3D) {
    using storage_info_t = storage_traits<backend_t>::storage_info_t<0, 3>;
    storage_info_t si(4, 5, 6);

    auto testee = make_storage_info_rt(si);

    EXPECT_THAT(testee.lengths(), ElementsAreArray(si.lengths()));
    EXPECT_THAT(testee.strides(), ElementsAreArray(si.strides()));
}

TEST(StorageInfoRT, Make3Dmasked) {
    using storage_info_t = storage_traits<backend_t>::special_storage_info_t<0, selector<1, 0, 1>>;
    storage_info_t si(4, 5, 6);

    auto testee = make_storage_info_rt(si);

    EXPECT_THAT(testee.lengths(), ElementsAreArray(si.lengths()));
    EXPECT_THAT(testee.strides(), ElementsAreArray(si.strides()));
}
