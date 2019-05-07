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

#include "../../common/cuda_util.hpp"
#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../extent.hpp"

#ifndef __CUDACC__
#error This is CUDA only header
#endif

namespace gridtools {
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
        template <int_t VBoundary>
        struct padded_boundary
            : std::integral_constant<int_t, VBoundary <= 1 ? 1 : VBoundary <= 2 ? 2 : VBoundary <= 4 ? 4 : 8> {
            GT_STATIC_ASSERT(VBoundary >= 0 && VBoundary <= 8, GT_INTERNAL_ERROR);
        };

        template <size_t NumThreads, int_t BlockSizeI, int_t BlockSizeJ, class Extent, class Fun>
        __global__ void __launch_bounds__(NumThreads) kernel(Fun const fun) {
            // jboundary_limit determines the number of warps required to execute (b,d,f)
            static constexpr auto jboundary_limit = BlockSizeJ + Extent::jplus::value - Extent::jminus::value;
            // iminus_limit adds to jboundary_limit an additional warp for regions (a,h,e)
            static constexpr auto iminus_limit = jboundary_limit + (Extent::iminus::value < 0 ? 1 : 0);
            // iminus_limit adds to iminus_limit an additional warp for regions (c,i,g)
            static constexpr auto iplus_limit = iminus_limit + (Extent::iplus::value > 0 ? 1 : 0);

            if (threadIdx.y < jboundary_limit) {
                fun(threadIdx.x, (int_t)threadIdx.y + Extent::jminus::value);
            } else if (threadIdx.y < iminus_limit) {
                static constexpr auto padded_boundary_ = padded_boundary<-Extent::iminus::value>::value;
                // we dedicate one warp to execute regions (a,h,e), so here we make sure we have enough threads
                GT_STATIC_ASSERT(jboundary_limit * padded_boundary_ <= BlockSizeI, GT_INTERNAL_ERROR);

                fun(-padded_boundary_ + (int_t)threadIdx.x % padded_boundary_,
                    (int_t)threadIdx.x / padded_boundary_ + Extent::jminus::value);
            } else {
                assert(threadIdx.y < iplus_limit);
                static constexpr auto padded_boundary_ = padded_boundary<Extent::iplus::value>::value;
                // we dedicate one warp to execute regions (c,i,g), so here we make sure we have enough threads
                GT_STATIC_ASSERT(jboundary_limit * padded_boundary_ <= BlockSizeI, GT_INTERNAL_ERROR);

                fun(threadIdx.x % padded_boundary_ + BlockSizeI,
                    (int_t)threadIdx.x / padded_boundary_ + Extent::jminus::value);
            }
        }
    } // namespace launch_kernel_impl_

    template <class Extent, int_t BlockSizeI, int_t BlockSizeJ, class Fun>
    GT_FORCE_INLINE void launch_kernel(dim3 const &blocks, Fun const &fun) {
        GT_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(std::is_trivially_copyable<Fun>::value, GT_INTERNAL_ERROR);

        static constexpr auto halo_lines = Extent::jplus::value - Extent::jminus::value +
                                           (Extent::iminus::value < 0 ? 1 : 0) + (Extent::iplus::value > 0 ? 1 : 0);

        static const size_t num_threads = BlockSizeI * (BlockSizeJ + halo_lines);

        launch_kernel_impl_::kernel<num_threads, BlockSizeI, BlockSizeJ, Extent>
            <<<blocks, dim3{BlockSizeI, BlockSizeJ + halo_lines, 1}>>>(fun);

#ifndef NDEBUG
        GT_CUDA_CHECK(cudaDeviceSynchronize());
#else
        GT_CUDA_CHECK(cudaGetLastError());
#endif
    }
} // namespace gridtools
