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
#include "../color.hpp"
#include "../dim.hpp"
#include "../extent.hpp"
#include "../sid/synthetic.hpp"
#include <memory>

namespace gridtools {
#ifndef GT_ICOSAHEDRAL_GRIDS
    namespace tmp_cuda_impl_ {
        template <class Strides, class BlockSizeI, class BlockSizeJ>
        Strides compute_strides(int_t n_blocks_i, int_t n_blocks_j) {
            return {integral_constant<int_t, 1>{}, // TODO support for default init {} in hymap
                BlockSizeI{},
                integral_constant<int_t, BlockSizeI{} * BlockSizeJ{}>{},
                GT_META_CALL(meta::at_c, (Strides, 2))::value * n_blocks_i,
                GT_META_CALL(meta::at_c, (Strides, 2))::value * n_blocks_i * n_blocks_j};
        }

        template <class BlockSizeI, class BlockSizeJ>
        std::size_t compute_size(int_t n_blocks_i, int_t n_blocks_j, int_t k_size) {
            return BlockSizeI{} * BlockSizeJ{} * n_blocks_i * n_blocks_j * k_size;
        }

        template <class Strides>
        int_t compute_origin_offset(Strides const &strides, int_t shift_i, int_t shift_j) {
            return shift_i * at_key<dim::i>(strides) + shift_j * at_key<dim::j>(strides);
        }
    } // namespace tmp_cuda_impl_

    namespace tmp_cuda {
        struct block_i;
        struct block_j;

        template <int_t, int_t>
        struct blocksize {};
    } // namespace tmp_cuda

    /**
     * @brief SID for CUDA temporaries.
     * get_origin() points to first element of compute domain
     * TODO(havogt): during integration we need to evaluate different types of alignment:
     *  - no alignment (current implementation)
     *  - alignment similar to old implementation (first data point in compute domain)
     *  - align first element of the temporary (i.e. in the redundant computation region)
     */
    template <class T,
        int_t ComputeBlockSizeI,
        int_t ComputeBlockSizeJ,
        int_t ExtentIMinus, // negative by convention
        int_t ExtentIPlus,
        int_t ExtentJMinus, // negative by convention
        int_t ExtentJPlus,
        class Allocator,
        class BlockSizeI = integral_constant<int_t, ComputeBlockSizeI - ExtentIMinus + ExtentIPlus>,
        class BlockSizeJ = integral_constant<int_t, ComputeBlockSizeJ - ExtentJMinus + ExtentJPlus>,
        class Strides = hymap::keys<dim::i, dim::j, tmp_cuda::block_i, tmp_cuda::block_j, dim::k>::values<
            integral_constant<int_t, 1>,
            BlockSizeI,
            integral_constant<int_t, BlockSizeI{} * BlockSizeJ{}>,
            int_t,
            int_t>>
    auto make_tmp_storage_cuda(tmp_cuda::blocksize<ComputeBlockSizeI, ComputeBlockSizeJ>,
        extent<ExtentIMinus, ExtentIPlus, ExtentJMinus, ExtentJPlus>,
        int_t n_blocks_i,
        int_t n_blocks_j,
        int_t k_size,
        Allocator &&alloc)
        GT_AUTO_RETURN(
            (sid::synthetic()
                    .set<sid::property::origin>(
                        alloc.template allocate<T>(
                            tmp_cuda_impl_::compute_size<BlockSizeI, BlockSizeJ>(n_blocks_i, n_blocks_j, k_size)) +
                        tmp_cuda_impl_::compute_origin_offset(
                            tmp_cuda_impl_::compute_strides<Strides, BlockSizeI, BlockSizeJ>(n_blocks_i, n_blocks_j),
                            -ExtentIMinus,
                            -ExtentJMinus))
                    .template set<sid::property::strides>(
                        tmp_cuda_impl_::compute_strides<Strides, BlockSizeI, BlockSizeJ>(n_blocks_i, n_blocks_j))
                    .template set<sid::property::ptr_diff, int_t>()
                    .template set<sid::property::strides_kind, Strides>()));

#else
#warning "correct branch"
    namespace tmp_cuda_impl_ {
        template <class Strides, class BlockSizeI, class BlockSizeJ>
        Strides compute_strides(int_t n_blocks_i, int_t n_blocks_j) {
            return {integral_constant<int_t, 1>{}, // TODO support for default init {} in hymap
                BlockSizeI{},
                integral_constant<int_t, BlockSizeI{} * BlockSizeJ{}>{},
                integral_constant<int_t, BlockSizeI{} * BlockSizeJ{} * 1>{}, // TODO 1 -> NColors
                GT_META_CALL(meta::at_c, (Strides, 2))::value * n_blocks_i,
                GT_META_CALL(meta::at_c, (Strides, 2))::value * n_blocks_i * n_blocks_j};
        }

        template <class BlockSizeI, class BlockSizeJ>
        std::size_t compute_size(int_t n_blocks_i, int_t n_blocks_j, int_t k_size) {
            return BlockSizeI{} * BlockSizeJ{} * n_blocks_i * n_blocks_j * k_size;
        }

        template <class Strides>
        int_t compute_origin_offset(Strides const &strides, int_t shift_i, int_t shift_j) {
            return shift_i * at_key<dim::i>(strides) + shift_j * at_key<dim::j>(strides);
        }
    } // namespace tmp_cuda_impl_

    namespace tmp_cuda {
        struct block_i;
        struct block_j;

        template <int_t, int_t>
        struct blocksize {};
    } // namespace tmp_cuda

    /**
     * @brief SID for CUDA temporaries.
     * get_origin() points to first element of compute domain
     * TODO(havogt): during integration we need to evaluate different types of alignment:
     *  - no alignment (current implementation)
     *  - alignment similar to old implementation (first data point in compute domain)
     *  - align first element of the temporary (i.e. in the redundant computation region)
     */
    template <class T,
        int_t ComputeBlockSizeI,
        int_t ComputeBlockSizeJ,
        int_t ExtentIMinus, // negative by convention
        int_t ExtentIPlus,
        int_t ExtentJMinus, // negative by convention
        int_t ExtentJPlus,
        uint_t NColors,
        class Allocator,
        class BlockSizeI = integral_constant<int_t, ComputeBlockSizeI - ExtentIMinus + ExtentIPlus>,
        class BlockSizeJ = integral_constant<int_t, ComputeBlockSizeJ - ExtentJMinus + ExtentJPlus>,
        class Strides = hymap::keys<dim::i, dim::j, dim::c, tmp_cuda::block_i, tmp_cuda::block_j, dim::k>::values<
            integral_constant<int_t, 1>,
            BlockSizeI,
            integral_constant<int_t, BlockSizeI{} * BlockSizeJ{}>,
            integral_constant<int_t, BlockSizeI{} * BlockSizeJ{} * NColors>,
            int_t,
            int_t>>
    auto make_tmp_storage_cuda(tmp_cuda::blocksize<ComputeBlockSizeI, ComputeBlockSizeJ>,
        extent<ExtentIMinus, ExtentIPlus, ExtentJMinus, ExtentJPlus>,
        color_type<NColors>,
        int_t n_blocks_i,
        int_t n_blocks_j,
        int_t k_size,
        Allocator &&alloc) {
        return sid::synthetic()
            .set<sid::property::origin>(
                alloc.template allocate<T>(
                    tmp_cuda_impl_::compute_size<BlockSizeI, BlockSizeJ>(n_blocks_i, n_blocks_j, k_size)) +
                tmp_cuda_impl_::compute_origin_offset(
                    tmp_cuda_impl_::compute_strides<Strides, BlockSizeI, BlockSizeJ>(n_blocks_i, n_blocks_j),
                    -ExtentIMinus,
                    -ExtentJMinus))
            .template set<sid::property::strides>(
                tmp_cuda_impl_::compute_strides<Strides, BlockSizeI, BlockSizeJ>(n_blocks_i, n_blocks_j))
            .template set<sid::property::ptr_diff, int_t>()
            .template set<sid::property::strides_kind, Strides>();
    }
#endif

} // namespace gridtools
