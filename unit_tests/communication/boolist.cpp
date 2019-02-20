/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "gtest/gtest.h"
#include <gridtools/common/boollist.hpp>
#include <gridtools/common/layout_map.hpp>
#include <iostream>
#include <stdlib.h>

TEST(Communication, boollist) {

    bool pass = true;

    for (int i = 0; i < 100000; ++i) {
        bool v0 = rand() % 2, v1 = rand() % 2, v2 = rand() % 2;

        gridtools::boollist<3> bl1(v0, v1, v2);

        if (bl1.value(0) != v0)
            pass = false;

        if (bl1.value(1) != v1)
            pass = false;

        if (bl1.value(2) != v2)
            pass = false;

        gridtools::boollist<3> bl2 = bl1.permute<gridtools::layout_map<1, 2, 0>>();

        if (bl2.value(0) != v2)
            pass = false;

        if (bl2.value(1) != v0)
            pass = false;

        if (bl2.value(2) != v1)
            pass = false;

        gridtools::boollist<3> bl3 = bl1.permute<gridtools::layout_map<2, 1, 0>>();

        if (bl3.value(0) != v2)
            pass = false;

        if (bl3.value(1) != v1)
            pass = false;

        if (bl3.value(2) != v0)
            pass = false;

        gridtools::boollist<3> bl4 = bl1.permute<gridtools::layout_map<0, 1, 2>>();

        if (bl4.value(0) != v0)
            pass = false;

        if (bl4.value(1) != v1)
            pass = false;

        if (bl4.value(2) != v2)
            pass = false;
    }

    EXPECT_TRUE(pass);
}
