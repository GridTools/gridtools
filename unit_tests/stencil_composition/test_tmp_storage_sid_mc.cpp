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

#include <gridtools/stencil_composition/extent.hpp>
#include <gridtools/stencil_composition/structured_grids/backend_mc/tmp_storage_sid.hpp>

using namespace gridtools;

TEST(tmp_storage_sid_mc, sid) {
    tmp_allocator_mc allocator;
    using extent_t = extent<-1, 2, 0, 3>;
    pos3<std::size_t> block_size{128, 1, 80};
    auto tmp = make_tmp_storage_mc<double, extent_t>(allocator, block_size);
}
