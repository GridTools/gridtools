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

#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple_util.hpp"
#include "../color.hpp"
#include "../dim.hpp"
#include "../extent.hpp"
#include "../sid/blocked_dim.hpp"
#include "../sid/contiguous.hpp"
#include "../sid/sid_shift_origin.hpp"

namespace gridtools {

    namespace tmp_cuda {
        template <int_t, int_t>
        struct blocksize {};
    } // namespace tmp_cuda

    namespace tmp_cuda_impl_ {
        template <class Extent>
        hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, -Extent::iminus::value>,
            integral_constant<int_t, -Extent::jminus::value>>
        origin_offset(Extent) {
            return {};
        }
    } // namespace tmp_cuda_impl_

#ifndef GT_ICOSAHEDRAL_GRIDS

    namespace tmp_cuda_impl_ {
        template <int_t ComputeBlockSizeI,
            int_t ComputeBlockSizeJ,
            int_t ExtentIMinus, // negative by convention
            int_t ExtentIPlus,
            int_t ExtentJMinus, // negative by convention
            int_t ExtentJPlus,
            int_t ExtentKMinus, // negative by convention
            int_t ExtentKPlus,
            class BlockSizeI = integral_constant<int_t, ComputeBlockSizeI - ExtentIMinus + ExtentIPlus>,
            class BlockSizeJ = integral_constant<int_t, ComputeBlockSizeJ - ExtentJMinus + ExtentJPlus>>
        auto sizes(tmp_cuda::blocksize<ComputeBlockSizeI, ComputeBlockSizeJ>,
            extent<ExtentIMinus, ExtentIPlus, ExtentJMinus, ExtentJPlus, ExtentKMinus, ExtentKPlus>,
            int_t n_blocks_i,
            int_t n_blocks_j,
            int_t k_size) {
            return tuple_util::make<
                hymap::keys<dim::i, dim::j, sid::blocked_dim<dim::i>, sid::blocked_dim<dim::j>, dim::k>::values>(
                BlockSizeI{}, BlockSizeJ{}, n_blocks_i, n_blocks_j, k_size);
        }
    } // namespace tmp_cuda_impl_

    /**
     * @brief SID for CUDA temporaries.
     * @param blocksize of the 2D CUDA block (same for all temporaries within a computation)
     * @param extent max extent of all temporaries in a computation (same for all temporaries within a computation)
     * @param n_blocks_i (same for all temporaries within a computation)
     * @param n_blocks_j (same for all temporaries within a computation)
     * @param k_size (can be different)
     * @param allocator
     *
     * get_origin() points to first element of compute domain
     * TODO(havogt): during integration we need to evaluate different types of alignment:
     *  - no alignment (current implementation)
     *  - alignment similar to old implementation (first data point in compute domain)
     *  - align first element of the temporary (i.e. in the redundant computation region)
     */
    template <class T, class ComputeBlockSize, class Extent, class Allocator>
    auto make_tmp_storage_cuda(ComputeBlockSize compute_block_size,
        Extent extent,
        int_t n_blocks_i,
        int_t n_blocks_j,
        int_t k_size,
        Allocator &alloc) {
        return sid::shift_sid_origin(
            sid::make_contiguous<T, int_t>(
                alloc, tmp_cuda_impl_::sizes(compute_block_size, extent, n_blocks_i, n_blocks_j, k_size)),
            tmp_cuda_impl_::origin_offset(extent));
    }

#else

    namespace tmp_cuda_impl_ {
        template <int_t ComputeBlockSizeI,
            int_t ComputeBlockSizeJ,
            int_t ExtentIMinus, // negative by convention
            int_t ExtentIPlus,
            int_t ExtentJMinus, // negative by convention
            int_t ExtentJPlus,
            int_t ExtentKMinus, // negative by convention
            int_t ExtentKPlus,
            uint_t NColors,
            class BlockSizeI = integral_constant<int_t, ComputeBlockSizeI - ExtentIMinus + ExtentIPlus>,
            class BlockSizeJ = integral_constant<int_t, ComputeBlockSizeJ - ExtentJMinus + ExtentJPlus>>
        auto sizes(tmp_cuda::blocksize<ComputeBlockSizeI, ComputeBlockSizeJ>,
            extent<ExtentIMinus, ExtentIPlus, ExtentJMinus, ExtentJPlus, ExtentKMinus, ExtentKPlus>,
            color_type<NColors>,
            int_t n_blocks_i,
            int_t n_blocks_j,
            int_t k_size) {
            return tuple_util::make<hymap::
                    keys<dim::i, dim::j, dim::c, sid::blocked_dim<dim::i>, sid::blocked_dim<dim::j>, dim::k>::values>(
                BlockSizeI{}, BlockSizeJ{}, integral_constant<int_t, NColors>{}, n_blocks_i, n_blocks_j, k_size);
        }
    } // namespace tmp_cuda_impl_

    /**
     * @brief SID for CUDA temporaries.
     * get_origin() points to first element of compute domain
     * TODO(havogt): during integration we need to evaluate different types of alignment:
     *  - no alignment (current implementation)
     *  - alignment similar to old implementation (first data point in compute domain)
     *  - align first element of the temporary (i.e. in the redundant computation region)
     */
    template <class T, class ComputeBlockSize, class Extent, class Color, class Allocator>
    auto make_tmp_storage_cuda(ComputeBlockSize compute_block_size,
        Extent extent,
        Color color,
        int_t n_blocks_i,
        int_t n_blocks_j,
        int_t k_size,
        Allocator &alloc) {
        return sid::shift_sid_origin(
            sid::make_contiguous<T, int_t>(
                alloc, tmp_cuda_impl_::sizes(compute_block_size, extent, color, n_blocks_i, n_blocks_j, k_size)),
            tmp_cuda_impl_::origin_offset(extent));
    }
#endif

} // namespace gridtools
