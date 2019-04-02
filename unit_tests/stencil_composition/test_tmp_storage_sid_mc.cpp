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
#include <gridtools/stencil_composition/sid/concept.hpp>
#include <gridtools/stencil_composition/structured_grids/backend_mc/tmp_storage_sid.hpp>

using namespace gridtools;

static constexpr std::size_t byte_alignment = 64;

TEST(tmp_storage_sid_mc, allocator) {
    tmp_allocator_mc allocator;

    std::size_t n = 100;
    auto ptr_holder = allocator.allocate<double>(n);

    double *ptr = ptr_holder();
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % byte_alignment, 0);

    for (std::size_t i = 0; i < n; ++i) {
        ptr[i] = 0;
        EXPECT_EQ(ptr[i], 0);
    }
}

TEST(tmp_storage_sid_mc, sid) {
    static constexpr std::size_t double_alignment = byte_alignment / sizeof(double);

    using extent_t = extent<-1, 2, -2, 3, -1, 2>;
    pos3<std::size_t> block_size{120, 2, 80};

    tmp_allocator_mc allocator;
    auto tmp = make_tmp_storage_mc<double, extent_t>(allocator, block_size);

    using tmp_t = decltype(tmp);

    static_assert(is_sid<tmp_t>(), "");
    static_assert(std::is_same<GT_META_CALL(sid::ptr_type, tmp_t), double *>(), "");

    double *ptr = sid::get_origin(tmp)();
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % byte_alignment, 0);

    auto strides = sid::get_strides(tmp);

    pos3<std::size_t> full_block_size{
        (block_size.i - extent_t::iminus::value + extent_t::iplus::value + double_alignment - 1) / double_alignment *
            double_alignment,
        block_size.j - extent_t::jminus::value + extent_t::jplus::value,
        block_size.k - extent_t::kminus::value + extent_t::kplus::value};

    EXPECT_EQ(at_key<dim::i>(strides), 1);
    EXPECT_EQ(at_key<dim::j>(strides), full_block_size.i * full_block_size.k);
    EXPECT_EQ(at_key<dim::k>(strides), full_block_size.i);
    EXPECT_EQ(at_key<thread_dim_mc>(strides), full_block_size.i * full_block_size.j * full_block_size.k);
}
