/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "shallow_water_enhanced.hpp"
#include "gtest/gtest.h"
#include <gridtools/tools/mpi_unit_test_driver/check_flags.hpp>

TEST(Communication, shallow_water_enhanced) {
    bool passed = shallow_water::test(8, 8, 1);
    EXPECT_TRUE(passed);
}
