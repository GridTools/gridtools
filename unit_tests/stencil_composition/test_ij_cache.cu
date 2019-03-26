/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/backend_cuda/ij_cache.hpp>

#include <gtest/gtest.h>

namespace {
    TEST(ij_cache, one_ij_cache) {
        gridtools::SharedAllocator allocator;
        gridtools::sid_ij_cache<int, 20, 4, 1, 1> ij_cache{allocator};
    }
} // namespace
