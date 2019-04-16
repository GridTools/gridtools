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

        template <typename T>
        class tmp_cuda_storage_sid : public ::testing::Test {
          public:
            using data_t = T;

            int_t n_blocks_i;
            int_t n_blocks_j;
            int_t k_size;

            //          private:
            simple_device_memory_allocator alloc;

          public:
            tmp_cuda_storage_sid() : n_blocks_i{11}, n_blocks_j{12}, k_size{13} {}

            auto get_tmp_storage()
                GT_AUTO_RETURN(make_tmp_storage_cuda<data_t>(tmp_cuda::blocksize<blocksize_i, blocksize_j>{},
                    extent<extent_i_minus, extent_i_plus, extent_j_minus, extent_j_plus>{},
                    n_blocks_i,
                    n_blocks_j,
                    k_size,
                    alloc));

            // test helper
            int stride0() const { return 1; }
            int stride1() const { return blocksize_i - extent_i_minus + extent_i_plus; }
            int stride2() const { return stride1() * (blocksize_j - extent_j_minus + extent_j_plus); }
            int stride3() const { return stride2() * n_blocks_i; }
            int stride4() const { return stride3() * n_blocks_j; }
            int extended_blocksize_i() const { return blocksize_i - extent_i_minus + extent_i_plus; }
            int extended_blocksize_j() const { return blocksize_j - extent_j_minus + extent_j_plus; }
            data_t *ptr_to_allocation() const { return static_cast<data_t *>(alloc.ptrs()[0].get()); }
            data_t *ptr_to_origin() const {
                return ptr_to_allocation() - stride0() * extent_i_minus - stride1() * extent_j_minus;
            }
            size_t n_elements() const {
                return tmp_cuda_impl_::compute_size<
                    std::integral_constant<int, blocksize_i - extent_i_minus + extent_i_plus>,
                    std::integral_constant<int, blocksize_j - extent_j_minus + extent_j_plus>>(
                    n_blocks_i, n_blocks_j, k_size);
            }
        };

        using tmp_cuda_storage_sid_float = tmp_cuda_storage_sid<float_type>;
        TEST_F(tmp_cuda_storage_sid_float, maker) {
            auto testee = get_tmp_storage();

            using tmp_cuda_t = decltype(testee);

            static_assert(sid::is_sid<tmp_cuda_t>::value, "is_sid()");

            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_type, tmp_cuda_t), data_t *>();
            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_diff_type, tmp_cuda_t), int_t>();

            auto strides = sid::get_strides(testee);

            EXPECT_EQ(stride0(), at_key<dim::i>(strides));
            EXPECT_EQ(stride1(), at_key<dim::j>(strides));
            EXPECT_EQ(stride2(), at_key<tmp_cuda::block_i>(strides));
            EXPECT_EQ(stride3(), at_key<tmp_cuda::block_j>(strides));
            EXPECT_EQ(stride4(), at_key<dim::k>(strides));

            EXPECT_EQ(ptr_to_origin(), sid::get_origin(testee)());
        }

        struct block_info {
            int i;
            int j;
        };

        template <class PtrHolder, class Strides>
        __global__ void write_block_index(PtrHolder ptr_holder, Strides strides) {
            int block_id_x = blockIdx.x;
            int block_id_y = blockIdx.y;
            int block_id_z = blockIdx.z;
            int thread_id_x = (int)threadIdx.x + extent_i_minus;
            int thread_id_y = (int)threadIdx.y + extent_j_minus;

            auto ptr = ptr_holder();
            sid::shift(ptr, device::at_key<dim::i>(strides), thread_id_x);
            sid::shift(ptr, device::at_key<dim::j>(strides), thread_id_y);
            sid::shift(ptr, device::at_key<tmp_cuda::block_i>(strides), block_id_x);
            sid::shift(ptr, device::at_key<tmp_cuda::block_j>(strides), block_id_y);
            sid::shift(ptr, device::at_key<dim::k>(strides), block_id_z);
            if (threadIdx.x >= -extent_i_minus && threadIdx.x < blockDim.x - extent_i_plus && //
                threadIdx.y >= -extent_j_minus && threadIdx.y < blockDim.y - extent_j_plus) {
                // in domain
                ptr->i = block_id_x;
                ptr->j = block_id_y;
            } else {
                // in redundant computation area
                ptr->i = -1;
                ptr->j = -1;
            }
        }

        using tmp_cuda_storage_sid_block = tmp_cuda_storage_sid<block_info>;
        TEST_F(tmp_cuda_storage_sid_block, write_in_blocks) {
            auto testee = get_tmp_storage();

            auto strides = sid::get_strides(testee);
            auto origin = sid::get_origin(testee);

            dim3 blocks(n_blocks_i, n_blocks_j, k_size);
            dim3 threads(extended_blocksize_i(), extended_blocksize_j(), 1);
            // fill domain area with blockid, redundant computation area is not touched
            write_block_index<<<blocks, threads>>>(origin, strides);
            GT_CUDA_CHECK(cudaDeviceSynchronize());

            data_t *result = new data_t[n_elements()];
            GT_CUDA_CHECK(
                cudaMemcpy(result, ptr_to_allocation(), n_elements() * sizeof(data_t), cudaMemcpyDeviceToHost));

            for (int i = extent_i_minus; i < blocksize_i + extent_i_plus; ++i)
                for (int j = extent_j_minus; j < blocksize_j + extent_j_plus; ++j)
                    for (size_t bi = 0; bi < n_blocks_i; ++bi)
                        for (size_t bj = 0; bj < n_blocks_j; ++bj)
                            for (size_t k = 0; k < k_size; ++k) {
                                auto ptr_to_origin = result - stride0() * extent_i_minus - stride1() * extent_j_minus;
                                auto stride =
                                    i * stride0() + j * stride1() + bi * stride2() + bj * stride3() + k * stride4();
                                if (i >= 0 && i < blocksize_i && j >= 0 && j < blocksize_j) {
                                    EXPECT_EQ(bi, ptr_to_origin[stride].i) << i << "/" << j;
                                    EXPECT_EQ(bj, ptr_to_origin[stride].j) << i << "/" << j;
                                } else {
                                    EXPECT_EQ(-1, ptr_to_origin[stride].i) << i << "/" << j;
                                    EXPECT_EQ(-1, ptr_to_origin[stride].i) << i << "/" << j;
                                }
                            }
        }
    } // namespace
} // namespace gridtools
