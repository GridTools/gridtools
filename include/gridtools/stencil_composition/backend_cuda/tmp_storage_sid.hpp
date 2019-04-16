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

    namespace tmp_cuda {
        struct block_i;
        struct block_j;

        template <int_t, int_t>
        struct blocksize {};
    } // namespace tmp_cuda

    namespace tmp_cuda_impl_ {
        template <class Strides>
        int_t origin_offset(Strides const &strides, int_t shift_i, int_t shift_j) {
            return shift_i * at_key<dim::i>(strides) + shift_j * at_key<dim::j>(strides);
        }

        template <class T, class Allocator>
        auto make_ptr_holder(Allocator &alloc, size_t num_elements)
            GT_AUTO_RETURN((allocate(alloc, meta::lazy::id<T>{}, num_elements)));

        template <class T, class Allocator, class Strides>
        auto make_origin(Allocator &alloc, size_t num_elements, Strides const &strides, int_t shift_i, int_t shift_j)
            GT_AUTO_RETURN((make_ptr_holder<T>(alloc, num_elements) + origin_offset(strides, shift_i, shift_j)));

        template <class Strides, class PtrHolder>
        auto make_synthetic(Strides const &strides, PtrHolder const &ptr)
            GT_AUTO_RETURN((sid::synthetic()
                                .set<sid::property::origin>(ptr)
                                .template set<sid::property::strides>(strides)
                                .template set<sid::property::ptr_diff, int_t>()
                                .template set<sid::property::strides_kind, Strides>()));

        template <class BlockSizeI, class BlockSizeJ, uint_t NColors = 1>
        std::size_t compute_size(int_t n_blocks_i, int_t n_blocks_j, int_t k_size) {
            return BlockSizeI::value * BlockSizeJ::value * NColors * n_blocks_i * n_blocks_j * k_size;
        }

    } // namespace tmp_cuda_impl_

#ifndef GT_ICOSAHEDRAL_GRIDS

    namespace tmp_cuda_impl_ {
        template <class BlockSizeI, class BlockSizeJ, class StrideI = integral_constant<int_t, 1>>
        hymap::keys<dim::i, dim::j, tmp_cuda::block_i, tmp_cuda::block_j, dim::k>::
            values<StrideI, BlockSizeI, integral_constant<int_t, BlockSizeI::value * BlockSizeJ::value>, int_t, int_t>
            compute_strides(int_t n_blocks_i, int_t n_blocks_j) {
            return {StrideI{}, // TODO support for default init {} in hymap
                BlockSizeI{},
                integral_constant<int_t, BlockSizeI::value * BlockSizeJ::value>{},
                BlockSizeI::value * BlockSizeJ::value * n_blocks_i,
                BlockSizeI::value * BlockSizeJ::value * n_blocks_i * n_blocks_j};
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
    template <class T,
        int_t ComputeBlockSizeI,
        int_t ComputeBlockSizeJ,
        int_t ExtentIMinus, // negative by convention
        int_t ExtentIPlus,
        int_t ExtentJMinus, // negative by convention
        int_t ExtentJPlus,
        class Allocator,
        class BlockSizeI = integral_constant<int_t, ComputeBlockSizeI - ExtentIMinus + ExtentIPlus>,
        class BlockSizeJ = integral_constant<int_t, ComputeBlockSizeJ - ExtentJMinus + ExtentJPlus>>
    auto make_tmp_storage_cuda(tmp_cuda::blocksize<ComputeBlockSizeI, ComputeBlockSizeJ>,
        extent<ExtentIMinus, ExtentIPlus, ExtentJMinus, ExtentJPlus, 0, 0>,
        int_t n_blocks_i,
        int_t n_blocks_j,
        int_t k_size,
        Allocator &alloc)
        GT_AUTO_RETURN((tmp_cuda_impl_::make_synthetic(
            tmp_cuda_impl_::compute_strides<BlockSizeI, BlockSizeJ>(n_blocks_i, n_blocks_j),
            tmp_cuda_impl_::make_origin<T>(alloc,
                tmp_cuda_impl_::compute_size<BlockSizeI, BlockSizeJ>(n_blocks_i, n_blocks_j, k_size),
                tmp_cuda_impl_::compute_strides<BlockSizeI, BlockSizeJ>(n_blocks_i, n_blocks_j),
                -ExtentIMinus,
                -ExtentJMinus))));

#else
    namespace tmp_cuda_impl_ {
        template <class Strides, class BlockSizeI, class BlockSizeJ, uint_t NColors>
        Strides compute_strides(int_t n_blocks_i, int_t n_blocks_j) {
            return Strides{integral_constant<int_t, 1>{}, // TODO support for default init {} in hymap
                BlockSizeI{},
                integral_constant<int_t, BlockSizeI{} * BlockSizeJ{}>{},
                integral_constant<int_t, BlockSizeI{} * BlockSizeJ{} * NColors>{},
                GT_META_CALL(meta::at_c, (Strides, 3))::value * n_blocks_i,
                GT_META_CALL(meta::at_c, (Strides, 3))::value * n_blocks_i * n_blocks_j};
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
            integral_constant<int_t, BlockSizeI::value * BlockSizeJ::value>,
            integral_constant<int_t, BlockSizeI::value * BlockSizeJ::value * NColors>,
            int_t,
            int_t>>
    auto make_tmp_storage_cuda(tmp_cuda::blocksize<ComputeBlockSizeI, ComputeBlockSizeJ>,
        extent<ExtentIMinus, ExtentIPlus, ExtentJMinus, ExtentJPlus>,
        color_type<NColors>,
        int_t n_blocks_i,
        int_t n_blocks_j,
        int_t k_size,
        Allocator &&alloc)
        GT_AUTO_RETURN((
            sid::synthetic()
                .set<sid::property::origin>(
                    alloc.template allocate<T>(
                        tmp_cuda_impl_::compute_size<BlockSizeI, BlockSizeJ>(n_blocks_i, n_blocks_j, k_size)) +
                    tmp_cuda_impl_::compute_origin_offset(
                        tmp_cuda_impl_::compute_strides<Strides, BlockSizeI, BlockSizeJ, NColors>(
                            n_blocks_i, n_blocks_j),
                        -ExtentIMinus,
                        -ExtentJMinus))
                .template set<sid::property::strides>(
                    tmp_cuda_impl_::compute_strides<Strides, BlockSizeI, BlockSizeJ, NColors>(n_blocks_i, n_blocks_j))
                .template set<sid::property::ptr_diff, int_t>()
                .template set<sid::property::strides_kind, Strides>()));
#endif

} // namespace gridtools
