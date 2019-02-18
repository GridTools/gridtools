/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "copy_stencil_parallel.hpp"
#include "gtest/gtest.h"
#include <gridtools/tools/mpi_unit_test_driver/check_flags.hpp>

TEST(copy_stencil_parallel, test) { EXPECT_TRUE(copy_stencil::test(13, 11, 7)); }
