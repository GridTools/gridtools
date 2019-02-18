/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "test_sequence_unpacker.hpp"
#include "gtest/gtest.h"

TEST(sequence_unpacker, test_unpack) {
    bool result = true;
    test_sequence_unpacker(&result);
    ASSERT_TRUE(result);
}
