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

#ifndef __CUDACC__
#error This is CUDA only header
#endif

#include <type_traits>

#include "../../common/cuda_util.hpp"
#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"
#include "../dim.hpp"
#include "../extent.hpp"

namespace gridtools {
    namespace cuda {
        /*
         *  In a typical cuda block we have the following regions
         *
         *    aa bbbbbbbb cc
         *    aa bbbbbbbb cc
         *
         *    hh dddddddd ii
         *    hh dddddddd ii
         *    hh dddddddd ii
         *    hh dddddddd ii
         *
         *    ee ffffffff gg
         *    ee ffffffff gg
         *
         * Regions b,d,f have warp (or multiple of warp size)
         * Size of regions a, c, h, i, e, g are determined by max_extent_t
         * Regions b,d,f are easily executed by dedicated warps (one warp for each line).
         * Regions (a,h,e) and (c,i,g) are executed by two specialized warp
         */

        namespace launch_kernel_impl_ {

            constexpr int_t ceil(int_t x) { return x < 2 ? 1 : 2 * ceil((x + 1) / 2); }

            template <class MaxExtent>
            struct extent_validator_f {
                GT_STATIC_ASSERT(is_extent<MaxExtent>::value, GT_INTERNAL_ERROR);

                int_t m_i_lo;
                int_t m_i_hi;
                int_t m_j_lo;
                int_t m_j_hi;

                GT_FUNCTION_DEVICE extent_validator_f(int_t i_pos, int_t j_pos, int_t i_block_size, int_t j_block_size)
                    : m_i_lo(i_pos), m_i_hi(i_pos - i_block_size), m_j_lo(j_pos), m_j_hi(j_pos - j_block_size) {}

                template <class Extent>
                GT_FUNCTION_DEVICE bool operator()(Extent) const {
                    GT_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
                    GT_STATIC_ASSERT(Extent::iminus::value >= MaxExtent::iminus::value, GT_INTERNAL_ERROR);
                    GT_STATIC_ASSERT(Extent::iplus::value <= MaxExtent::iplus::value, GT_INTERNAL_ERROR);
                    GT_STATIC_ASSERT(Extent::jminus::value >= MaxExtent::jminus::value, GT_INTERNAL_ERROR);
                    GT_STATIC_ASSERT(Extent::jplus::value <= MaxExtent::jplus::value, GT_INTERNAL_ERROR);

                    return Extent::iminus::value <= m_i_lo && Extent::iplus::value > m_i_hi &&
                           Extent::jminus::value <= m_j_lo && Extent::jplus::value > m_j_hi;
                }
            };

            template <size_t NumThreads, int_t BlockSizeI, int_t BlockSizeJ, class Extent, class Fun>
            __global__ void __launch_bounds__(NumThreads) wrapper(Fun const fun, int_t i_size, int_t j_size) {
                // jboundary_limit determines the number of warps required to execute (b,d,f)
                static constexpr auto jboundary_limit = BlockSizeJ + Extent::jplus::value - Extent::jminus::value;
                // iminus_limit adds to jboundary_limit an additional warp for regions (a,h,e)
                static constexpr auto iminus_limit = jboundary_limit + (Extent::iminus::value < 0 ? 1 : 0);

                int_t i_block, j_block;

                if (threadIdx.y < jboundary_limit) {
                    i_block = (int_t)threadIdx.x;
                    j_block = (int_t)threadIdx.y + Extent::jminus::value;
                } else if (threadIdx.y < iminus_limit) {
                    assert(Extent::iminus::value < 0);
                    static constexpr auto boundary = ceil(-Extent::iminus::value);
                    // we dedicate one warp to execute regions (a,h,e), so here we make sure we have enough threads
                    GT_STATIC_ASSERT(jboundary_limit * boundary <= BlockSizeI, GT_INTERNAL_ERROR);

                    i_block = -boundary + (int_t)threadIdx.x % boundary;
                    j_block = (int_t)threadIdx.x / boundary + Extent::jminus::value;
                } else {
                    assert(Extent::iplus::value > 0);
                    assert(threadIdx.y < iminus_limit + 1);
                    static constexpr auto boundary = ceil(Extent::iplus::value);
                    // we dedicate one warp to execute regions (c,i,g), so here we make sure we have enough threads
                    GT_STATIC_ASSERT(jboundary_limit * boundary <= BlockSizeI, GT_INTERNAL_ERROR);

                    i_block = (int_t)threadIdx.x % boundary + BlockSizeI;
                    j_block = (int_t)threadIdx.x / boundary + Extent::jminus::value;
                }

                int_t i_block_size =
                    (blockIdx.x + 1) * BlockSizeI < i_size ? BlockSizeI : i_size - blockIdx.x * BlockSizeI;
                int_t j_block_size =
                    (blockIdx.y + 1) * BlockSizeJ < j_size ? BlockSizeJ : j_size - blockIdx.y * BlockSizeJ;

                fun(i_block, j_block, extent_validator_f<Extent>{i_block, j_block, i_block_size, j_block_size});
            }

            template <size_t NumThreads, int_t BlockSizeI, int_t BlockSizeJ, class Fun>
            __global__ void __launch_bounds__(NumThreads)
                zero_extent_wrapper(Fun const fun, int_t i_size, int_t j_size) {
                if (blockIdx.x * BlockSizeI + threadIdx.x < i_size && blockIdx.y * BlockSizeJ + threadIdx.y < j_size)
                    fun(threadIdx.x, threadIdx.y, [](auto) { return true; });
                else
                    fun(threadIdx.x, threadIdx.y, [](auto) { return false; });
            }

            template <size_t NumThreads, class Fun>
            __global__ void __launch_bounds__(NumThreads) trivial_wrapper(Fun const fun) {
                fun(threadIdx.x, threadIdx.y, [](auto) { return true; });
            }

            template <class Extent>
            constexpr bool is_empty_ij_extents() {
                return Extent::iminus::value == 0 && Extent::iplus::value == 0 && Extent::jminus::value == 0 &&
                       Extent::jplus::value == 0;
            }

        } // namespace launch_kernel_impl_

        template <class Extent,
            int_t BlockSizeI,
            int_t BlockSizeJ,
            class Fun,
            std::enable_if_t<!launch_kernel_impl_::is_empty_ij_extents<Extent>(), int> = 0>
        GT_FORCE_INLINE void launch_kernel(
            int_t i_size, int_t j_size, uint_t zblocks, Fun const &fun, size_t shared_memory_size = 0) {
            GT_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(Extent::iminus::value <= 0, GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(Extent::iplus::value >= 0, GT_INTERNAL_ERROR);

            GT_STATIC_ASSERT(std::is_trivially_copyable<Fun>::value, GT_INTERNAL_ERROR);

            static constexpr auto halo_lines = Extent::jplus::value - Extent::jminus::value +
                                               (Extent::iminus::value < 0 ? 1 : 0) + (Extent::iplus::value > 0 ? 1 : 0);
            static const size_t num_threads = BlockSizeI * (BlockSizeJ + halo_lines);

            uint_t xblocks = (i_size + BlockSizeI - 1) / BlockSizeI;
            uint_t yblocks = (j_size + BlockSizeJ - 1) / BlockSizeJ;

            dim3 blocks = {xblocks, yblocks, zblocks};
            dim3 threads = {BlockSizeI, BlockSizeJ + halo_lines, 1};

            launch_kernel_impl_::wrapper<num_threads, BlockSizeI, BlockSizeJ, Extent>
                <<<blocks, threads, shared_memory_size>>>(fun, i_size, j_size);

#ifndef NDEBUG
            GT_CUDA_CHECK(cudaDeviceSynchronize());
#else
            GT_CUDA_CHECK(cudaGetLastError());
#endif
        }

        template <class Extent,
            int_t BlockSizeI,
            int_t BlockSizeJ,
            class Fun,
            std::enable_if_t<launch_kernel_impl_::is_empty_ij_extents<Extent>(), int> = 0>
        GT_FORCE_INLINE void launch_kernel(
            int_t i_size, int_t j_size, uint_t zblocks, Fun const &fun, size_t shared_memory_size = 0) {

            GT_STATIC_ASSERT(std::is_trivially_copyable<Fun>::value, GT_INTERNAL_ERROR);

            static const size_t num_threads = BlockSizeI * BlockSizeJ;

            uint_t xblocks = (i_size + BlockSizeI - 1) / BlockSizeI;
            uint_t yblocks = (j_size + BlockSizeJ - 1) / BlockSizeJ;

            dim3 blocks = {xblocks, yblocks, zblocks};
            dim3 threads = {BlockSizeI, BlockSizeJ, 1};

            if (i_size % BlockSizeI == 0 && j_size % BlockSizeJ == 0)
                launch_kernel_impl_::trivial_wrapper<num_threads><<<blocks, threads, shared_memory_size>>>(fun);
            else
                launch_kernel_impl_::zero_extent_wrapper<num_threads, BlockSizeI, BlockSizeJ>
                    <<<blocks, threads, shared_memory_size>>>(fun, i_size, j_size);

#ifndef NDEBUG
            GT_CUDA_CHECK(cudaDeviceSynchronize());
#else
            GT_CUDA_CHECK(cudaGetLastError());
#endif
        }
    } // namespace cuda
} // namespace gridtools
