/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/backend_cuda/simple_device_memory_allocator.hpp>
#include <gridtools/stencil_composition/backend_cuda/tmp_storage_sid.hpp>
#include <gridtools/stencil_composition/sid/concept.hpp>
#include <gridtools/tools/backend_select.hpp>

#include <type_traits>

#include "../../cuda_test_helper.hpp"
#include "../../test_helper.hpp"
#include <gtest/gtest.h>

namespace gridtools {
    namespace {

        constexpr int_t extent_i_minus = -1;
        constexpr int_t extent_i_plus = 2;
        constexpr int_t extent_j_minus = -3;
        constexpr int_t extent_j_plus = 4;

        constexpr int_t blocksize_i = 32;
        constexpr int_t blocksize_j = 8;

        constexpr uint_t n_colors = 2;

        template <typename T>
        class tmp_cuda_storage_sid : public ::testing::Test {
          public:
            using data_t = T;

            int_t n_blocks_i;
            int_t n_blocks_j;
            int_t k_size;

          private:
            simple_device_memory_allocator alloc;

          public:
            tmp_cuda_storage_sid() : n_blocks_i{11}, n_blocks_j{12}, k_size{13} {}

            auto get_tmp_storage()
                GT_AUTO_RETURN(make_tmp_storage_cuda<data_t>(tmp_cuda::blocksize<blocksize_i, blocksize_j>{},
                    extent<extent_i_minus, extent_i_plus, extent_j_minus, extent_j_plus>{},
                    color_type<n_colors>{},
                    n_blocks_i,
                    n_blocks_j,
                    k_size,
                    alloc));

            // test helper
            int stride0() const { return 1; }
            int stride1() const { return blocksize_i - extent_i_minus + extent_i_plus; }
            int stride2() const { return stride1() * (blocksize_j - extent_j_minus + extent_j_plus); }
            int stride3() const { return stride2() * n_colors; }
            int stride4() const { return stride3() * n_blocks_i; }
            int stride5() const { return stride4() * n_blocks_j; }
            data_t *ptr_to_allocation() const { return static_cast<data_t *>(alloc.ptrs()[0].get()); }
            data_t *ptr_to_origin() const {
                return ptr_to_allocation() - stride0() * extent_i_minus - stride1() * extent_j_minus;
            }
        };

        using tmp_cuda_storage_sid_float = tmp_cuda_storage_sid<float_type>;
        TEST_F(tmp_cuda_storage_sid_float, maker) {
            auto testee = get_tmp_storage();

            using tmp_cuda_t = decltype(testee);
            static_assert(is_sid<tmp_cuda_t>(), "");
            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_type, tmp_cuda_t), data_t *>();
            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_diff_type, tmp_cuda_t), int_t>();

            auto strides = sid::get_strides(testee);

            EXPECT_EQ(stride0(), at_key<dim::i>(strides));
            EXPECT_EQ(stride1(), at_key<dim::j>(strides));
            EXPECT_EQ(stride2(), at_key<dim::c>(strides));
            EXPECT_EQ(stride3(), at_key<tmp_cuda::block_i>(strides));
            EXPECT_EQ(stride4(), at_key<tmp_cuda::block_j>(strides));
            EXPECT_EQ(stride5(), at_key<dim::k>(strides));

            EXPECT_EQ(ptr_to_origin(), sid::get_origin(testee)());
        }

    } // namespace
} // namespace gridtools
