/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/backend_cuda/tmp_storage_sid.hpp>

#include <gridtools/stencil_composition/sid/concept.hpp>
#include <gridtools/stencil_composition/sid/simple_ptr_holder.hpp>
#include <gridtools/tools/backend_select.hpp>

#include <memory>
#include <type_traits>
#include <vector>

#include "../tools/multiplet.hpp"
#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        // Strided access can be tested with host memory
        class simple_host_memory_allocator {
            std::vector<std::shared_ptr<void>> m_ptrs;

          public:
            template <class T>
            sid::host::simple_ptr_holder<T *> allocate(size_t num_elements) {
                T *ptr = new T[num_elements];
                m_ptrs.emplace_back(ptr, [](T *ptr) { delete[] ptr; });
                return {ptr};
            }
        };

        template <class Tag, class T = typename Tag::type>
        sid::host::simple_ptr_holder<T *> allocate(simple_host_memory_allocator &alloc, Tag, size_t num_elements) {
            return alloc.template allocate<T>(num_elements);
        }

        constexpr int_t extent_i_minus = -1;
        constexpr int_t extent_i_plus = 2;
        constexpr int_t extent_j_minus = -3;
        constexpr int_t extent_j_plus = 4;

        constexpr int_t blocksize_i = 32;
        constexpr int_t blocksize_j = 8;

        int_t n_blocks_i = 11;
        int_t n_blocks_j = 12;
        int_t k_size = 13;

#ifndef GT_ICOSAHEDRAL_GRIDS
        TEST(tmp_cuda_storage_sid_block, write_in_blocks) {
            using index_info = multiplet<5>;
            simple_host_memory_allocator alloc;

            auto testee = make_tmp_storage_cuda<index_info>(tmp_cuda::blocksize<blocksize_i, blocksize_j>{},
                extent<extent_i_minus, extent_i_plus, extent_j_minus, extent_j_plus>{},
                n_blocks_i,
                n_blocks_j,
                k_size,
                alloc);

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
                                sid::shift(ptr, host::at_key<sid::blocked_dim<dim::i>>(strides), bi);
                                sid::shift(ptr, host::at_key<sid::blocked_dim<dim::j>>(strides), bj);
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
                                sid::shift(ptr, host::at_key<sid::blocked_dim<dim::i>>(strides), bi);
                                sid::shift(ptr, host::at_key<sid::blocked_dim<dim::j>>(strides), bj);
                                sid::shift(ptr, host::at_key<dim::k>(strides), k);
                                ASSERT_EQ((index_info{i, j, bi, bj, k}), *ptr);
                            }
        }
#else
        constexpr int_t ncolors = 2;
        TEST(tmp_cuda_storage_sid_block, write_in_blocks) {
            using index_info = multiplet<6>;
            simple_host_memory_allocator alloc;

            auto testee = make_tmp_storage_cuda<index_info>(tmp_cuda::blocksize<blocksize_i, blocksize_j>{},
                extent<extent_i_minus, extent_i_plus, extent_j_minus, extent_j_plus>{},
                color_type<ncolors>{},
                n_blocks_i,
                n_blocks_j,
                k_size,
                alloc);

            auto strides = sid::get_strides(testee);
            auto origin = sid::get_origin(testee);

            // write block id
            for (int i = extent_i_minus; i < blocksize_i + extent_i_plus; ++i)
                for (int j = extent_j_minus; j < blocksize_j + extent_j_plus; ++j)
                    for (int c = 0; c < ncolors; ++c)
                        for (int bi = 0; bi < n_blocks_i; ++bi)
                            for (int bj = 0; bj < n_blocks_j; ++bj)
                                for (int k = 0; k < k_size; ++k) {
                                    auto ptr = origin();
                                    sid::shift(ptr, host::at_key<dim::i>(strides), i);
                                    sid::shift(ptr, host::at_key<dim::j>(strides), j);
                                    sid::shift(ptr, host::at_key<dim::c>(strides), c);
                                    sid::shift(ptr, host::at_key<sid::blocked_dim<dim::i>>(strides), bi);
                                    sid::shift(ptr, host::at_key<sid::blocked_dim<dim::j>>(strides), bj);
                                    sid::shift(ptr, host::at_key<dim::k>(strides), k);
                                    *ptr = {i, j, c, bi, bj, k};
                                }

            // validate that block id is correct, i.e. there were no overlapping memory accesses in the write
            for (int i = extent_i_minus; i < blocksize_i + extent_i_plus; ++i)
                for (int j = extent_j_minus; j < blocksize_j + extent_j_plus; ++j)
                    for (int c = 0; c < ncolors; ++c)
                        for (int bi = 0; bi < n_blocks_i; ++bi)
                            for (int bj = 0; bj < n_blocks_j; ++bj)
                                for (int k = 0; k < k_size; ++k) {
                                    auto ptr = origin();
                                    sid::shift(ptr, host::at_key<dim::i>(strides), i);
                                    sid::shift(ptr, host::at_key<dim::j>(strides), j);
                                    sid::shift(ptr, host::at_key<dim::c>(strides), c);
                                    sid::shift(ptr, host::at_key<sid::blocked_dim<dim::i>>(strides), bi);
                                    sid::shift(ptr, host::at_key<sid::blocked_dim<dim::j>>(strides), bj);
                                    sid::shift(ptr, host::at_key<dim::k>(strides), k);
                                    ASSERT_EQ((index_info{i, j, c, bi, bj, k}), *ptr);
                                }
        }
#endif
    } // namespace
} // namespace gridtools
