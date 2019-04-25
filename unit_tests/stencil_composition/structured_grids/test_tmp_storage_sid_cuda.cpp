/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../test_tmp_storage_sid_cuda.hpp"

#include <gridtools/stencil_composition/backend_cuda/tmp_storage_sid.hpp>
#include <gridtools/stencil_composition/sid/concept.hpp>
#include <gridtools/tools/backend_select.hpp>

#include <type_traits>

#include "../../tools/multiplet.hpp"

#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        constexpr int_t extent_i_minus = -1;
        constexpr int_t extent_i_plus = 2;
        constexpr int_t extent_j_minus = -3;
        constexpr int_t extent_j_plus = 4;

        constexpr int_t blocksize_i = 32;
        constexpr int_t blocksize_j = 8;

        template <typename T, typename Allocator> // test fixture works with default constructible allocators
        class tmp_cuda_storage_sid : public ::testing::Test {
          public:
            using data_t = T;

            int_t n_blocks_i;
            int_t n_blocks_j;
            int_t k_size;

          private:
            Allocator alloc;

          public:
            tmp_cuda_storage_sid() : n_blocks_i{11}, n_blocks_j{12}, k_size{13} {}

            auto get_tmp_storage()
                GT_AUTO_RETURN(make_tmp_storage_cuda<data_t>(tmp_cuda::blocksize<blocksize_i, blocksize_j>{},
                    extent<extent_i_minus, extent_i_plus, extent_j_minus, extent_j_plus>{},
                    n_blocks_i,
                    n_blocks_j,
                    k_size,
                    alloc));
        };

        // Strided access can be tested with host memory
        using index_info = multiplet<5>;
        using tmp_cuda_storage_sid_block = tmp_cuda_storage_sid<index_info, simple_host_memory_allocator>;
        TEST_F(tmp_cuda_storage_sid_block, write_in_blocks) {

            auto testee = get_tmp_storage();

            auto strides = sid::get_strides(testee);
            auto origin = sid::get_origin(testee);

            // write block id
            for (int i = extent_i_minus; i < blocksize_i + extent_i_plus; ++i)
                for (int j = extent_j_minus; j < blocksize_j + extent_j_plus; ++j)
                    for (int bi = 0; bi < n_blocks_i; ++bi)
                        for (int bj = 0; bj < n_blocks_j; ++bj)
                            for (int k = 0; k < k_size; ++k) {
                                auto ptr = origin();
                                sid::shift(ptr, host::at_key<dim::i>(strides), i);
                                sid::shift(ptr, host::at_key<dim::j>(strides), j);
                                sid::shift(ptr, host::at_key<tmp_cuda::block_i>(strides), bi);
                                sid::shift(ptr, host::at_key<tmp_cuda::block_j>(strides), bj);
                                sid::shift(ptr, host::at_key<dim::k>(strides), k);
                                *ptr = {i, j, bi, bj, k};
                            }

            // validate that block id is correct, i.e. there were no overlapping memory accesses in the write
            for (int i = extent_i_minus; i < blocksize_i + extent_i_plus; ++i)
                for (int j = extent_j_minus; j < blocksize_j + extent_j_plus; ++j)
                    for (int bi = 0; bi < n_blocks_i; ++bi)
                        for (int bj = 0; bj < n_blocks_j; ++bj)
                            for (int k = 0; k < k_size; ++k) {
                                auto ptr = origin();
                                sid::shift(ptr, host::at_key<dim::i>(strides), i);
                                sid::shift(ptr, host::at_key<dim::j>(strides), j);
                                sid::shift(ptr, host::at_key<tmp_cuda::block_i>(strides), bi);
                                sid::shift(ptr, host::at_key<tmp_cuda::block_j>(strides), bj);
                                sid::shift(ptr, host::at_key<dim::k>(strides), k);
                                ASSERT_EQ((index_info{i, j, bi, bj, k}), *ptr);
                            }
        }
    } // namespace
} // namespace gridtools
