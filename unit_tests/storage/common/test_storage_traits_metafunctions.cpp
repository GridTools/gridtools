/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "gtest/gtest.h"

#include <type_traits>

#include <gridtools/common/gt_assert.hpp>
#include <gridtools/storage/common/storage_traits_metafunctions.hpp>

using namespace gridtools;

TEST(StorageTraitsMetafunctions, CudaLayout) {
    // 3D
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<2, 1, 0>, selector<1, 0, 0>>::type,
                         layout_map<0, -1, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<2, 1, 0>, selector<0, 1, 0>>::type,
                         layout_map<-1, 0, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<2, 1, 0>, selector<0, 0, 1>>::type,
                         layout_map<-1, -1, 0>>::type::value),
        "");

    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<2, 1, 0>, selector<1, 1, 0>>::type,
                         layout_map<1, 0, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<2, 1, 0>, selector<0, 1, 1>>::type,
                         layout_map<-1, 1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<2, 1, 0>, selector<1, 0, 1>>::type,
                         layout_map<1, -1, 0>>::type::value),
        "");

    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<2, 1, 0>, selector<1, 1, 1>>::type,
                         layout_map<2, 1, 0>>::type::value),
        "");

    // 4D
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<0, 0, 0, 1>>::type,
                         layout_map<-1, -1, -1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<0, 0, 1, 0>>::type,
                         layout_map<-1, -1, 0, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<0, 1, 0, 0>>::type,
                         layout_map<-1, 0, -1, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<1, 0, 0, 0>>::type,
                         layout_map<0, -1, -1, -1>>::type::value),
        "");

    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<0, 0, 1, 1>>::type,
                         layout_map<-1, -1, 1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<0, 1, 1, 0>>::type,
                         layout_map<-1, 1, 0, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<1, 1, 0, 0>>::type,
                         layout_map<1, 0, -1, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<1, 0, 0, 1>>::type,
                         layout_map<1, -1, -1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<0, 1, 0, 1>>::type,
                         layout_map<-1, 1, -1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<1, 0, 1, 0>>::type,
                         layout_map<1, -1, 0, -1>>::type::value),
        "");

    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<0, 1, 1, 1>>::type,
                         layout_map<-1, 2, 1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<1, 1, 1, 0>>::type,
                         layout_map<2, 1, 0, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<1, 1, 0, 1>>::type,
                         layout_map<2, 1, -1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<1, 0, 1, 1>>::type,
                         layout_map<2, -1, 1, 0>>::type::value),
        "");

    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<3, 2, 1, 0>, selector<1, 1, 1, 1>>::type,
                         layout_map<3, 2, 1, 0>>::type::value),
        "");
}

TEST(StorageTraitsMetafunctions, HostLayout) {
    // 3D
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<0, 1, 2>, selector<1, 0, 0>>::type,
                         layout_map<0, -1, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<0, 1, 2>, selector<0, 1, 0>>::type,
                         layout_map<-1, 0, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<0, 1, 2>, selector<0, 0, 1>>::type,
                         layout_map<-1, -1, 0>>::type::value),
        "");

    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<0, 1, 2>, selector<1, 1, 0>>::type,
                         layout_map<0, 1, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<0, 1, 2>, selector<0, 1, 1>>::type,
                         layout_map<-1, 0, 1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<0, 1, 2>, selector<1, 0, 1>>::type,
                         layout_map<0, -1, 1>>::type::value),
        "");

    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<0, 1, 2>, selector<1, 1, 1>>::type,
                         layout_map<0, 1, 2>>::type::value),
        "");

    // 4D
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<0, 0, 0, 1>>::type,
                         layout_map<-1, -1, -1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<0, 0, 1, 0>>::type,
                         layout_map<-1, -1, 0, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<0, 1, 0, 0>>::type,
                         layout_map<-1, 0, -1, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<1, 0, 0, 0>>::type,
                         layout_map<0, -1, -1, -1>>::type::value),
        "");

    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<0, 0, 1, 1>>::type,
                         layout_map<-1, -1, 1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<0, 1, 1, 0>>::type,
                         layout_map<-1, 0, 1, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<1, 1, 0, 0>>::type,
                         layout_map<0, 1, -1, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<1, 0, 0, 1>>::type,
                         layout_map<1, -1, -1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<0, 1, 0, 1>>::type,
                         layout_map<-1, 1, -1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<1, 0, 1, 0>>::type,
                         layout_map<0, -1, 1, -1>>::type::value),
        "");

    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<0, 1, 1, 1>>::type,
                         layout_map<-1, 1, 2, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<1, 1, 1, 0>>::type,
                         layout_map<0, 1, 2, -1>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<1, 1, 0, 1>>::type,
                         layout_map<1, 2, -1, 0>>::type::value),
        "");
    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<1, 0, 1, 1>>::type,
                         layout_map<1, -1, 2, 0>>::type::value),
        "");

    GT_STATIC_ASSERT((std::is_same<typename get_special_layout<layout_map<1, 2, 3, 0>, selector<1, 1, 1, 1>>::type,
                         layout_map<1, 2, 3, 0>>::type::value),
        "");
}
