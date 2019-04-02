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

TEST(tmp_storage_sid_mc, sid) {
    tmp_allocator_mc allocator;
    using extent_t = extent<-1, 2, 0, 3, -1, 0>;
    pos3<std::size_t> block_size{125, 2, 80};
    auto tmp = make_tmp_storage_mc<double, extent_t>(allocator, block_size);

    using tmp_t = decltype(tmp);

    static_assert(is_sid<tmp_t>(), "");

    static_assert(std::is_same<GT_META_CALL(sid::ptr_type, tmp_t), double *>(), "");

    using strides_t =
        hymap::keys<dim::i, dim::k, dim::j, thread_dim>::values<integral_constant<int, 1>, int_t, int_t, int_t>;
    static_assert(std::is_same<GT_META_CALL(sid::strides_type, tmp_t), strides_t>(), "");

    auto strides = sid::get_strides(tmp);
    static_assert(decay_t<decltype(at_key<dim::i>(strides))>::value == 1, "");
    EXPECT_EQ(at_key<dim::j>(strides), 128 * 81);
    EXPECT_EQ(at_key<dim::k>(strides), 128);
    EXPECT_EQ(at_key<thread_dim>(strides), 128 * 5 * 81);
}
