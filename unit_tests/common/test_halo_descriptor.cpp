/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "gtest/gtest.h"
#include <gridtools/common/defs.hpp>
#include <gridtools/common/halo_descriptor.hpp>

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
