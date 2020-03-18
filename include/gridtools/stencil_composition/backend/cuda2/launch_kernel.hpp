/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>
#include <utility>

#include "../../../common/cuda_util.hpp"
#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"
#include "../../common/dim.hpp"
#include "../../common/extent.hpp"

namespace gridtools {
    namespace cuda2 {
        namespace launch_kernel_impl_ {
            template <class MaxExtent>
            struct extent_validator_f {
                static_assert(is_extent<MaxExtent>::value, GT_INTERNAL_ERROR);

                int_t m_i_lo;
                int_t m_i_hi;
                int_t m_j_block_size;

                GT_FUNCTION_DEVICE extent_validator_f(int_t i_pos, int_t i_block_size, int_t j_block_size)
                    : m_i_lo(i_pos), m_i_hi(i_pos - i_block_size), m_j_block_size(j_block_size) {}

                template <class Extent, class I>
                GT_FUNCTION_DEVICE bool operator()(Extent, I i, int_t j) const {
                    static_assert(is_extent<Extent>::value, GT_INTERNAL_ERROR);
                    static_assert(Extent::iminus::value >= MaxExtent::iminus::value, GT_INTERNAL_ERROR);
                    static_assert(Extent::iplus::value <= MaxExtent::iplus::value, GT_INTERNAL_ERROR);
                    static_assert(Extent::jminus::value >= MaxExtent::jminus::value, GT_INTERNAL_ERROR);
                    static_assert(Extent::jplus::value <= MaxExtent::jplus::value, GT_INTERNAL_ERROR);

                    return Extent::minus(dim::i()) - i <= m_i_lo && Extent::plus(dim::i()) - i > m_i_hi &&
                           Extent::jminus::value <= j && Extent::jplus::value > j - m_j_block_size;
                }
            };

            template <size_t NumThreads, int_t BlockSizeI, int_t BlockSizeJ, class Extent, class Fun>
            __global__ void __launch_bounds__(NumThreads) wrapper(Fun const fun, int_t i_size, int_t j_size) {
                int_t i_block = threadIdx.x;
                if (i_block >= BlockSizeI + Extent::plus(dim::i()))
                    i_block -= Extent::extend(dim::i(), BlockSizeI);

                int_t i_block_size =
                    (blockIdx.x + 1) * BlockSizeI < i_size ? BlockSizeI : i_size - blockIdx.x * BlockSizeI;
                int_t j_block_size =
                    (blockIdx.y + 1) * BlockSizeJ < j_size ? BlockSizeJ : j_size - blockIdx.y * BlockSizeJ;

                fun(i_block, extent_validator_f<Extent>{i_block, i_block_size, j_block_size});
            }

            template <class Extent, int_t BlockSizeI, int_t BlockSizeJ, class Fun>
            void launch_kernel(int_t i_size, int_t j_size, uint_t zblocks, Fun fun, size_t shared_memory_size = 0) {
                static_assert(is_extent<Extent>::value, GT_INTERNAL_ERROR);
                static_assert(Extent::iminus::value <= 0, GT_INTERNAL_ERROR);
                static_assert(Extent::iplus::value >= 0, GT_INTERNAL_ERROR);
                static_assert(std::is_trivially_copyable<Fun>::value, GT_INTERNAL_ERROR);

                static constexpr size_t num_threads = BlockSizeI + size_t(Extent::iplus::value - Extent::iminus::value);

                cuda_util::launch(
                    dim3((i_size + BlockSizeI - 1) / BlockSizeI, (j_size + BlockSizeJ - 1) / BlockSizeJ, zblocks),
                    dim3(Extent::extend(dim::i(), BlockSizeI), 1, 1),
                    shared_memory_size,
                    wrapper<num_threads, BlockSizeI, BlockSizeJ, Extent, Fun>,
                    std::move(fun),
                    i_size,
                    j_size);
            }
        } // namespace launch_kernel_impl_

        using launch_kernel_impl_::launch_kernel;
    } // namespace cuda2
} // namespace gridtools
