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

#include "../../common/cuda_allocator.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
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
                    .set<sid::property::origin>(host_device::constant<T *>{
                        alloc.template allocate<T>(BlockSizeI{} * BlockSizeJ{} * n_blocks_i * n_blocks_j * k_size)() -
                        ExtentIMinus * GT_META_CALL(meta::at_c, (Strides, 0)){}
                        // TODO access with dim::i, e.g. mp_find<Strides, dim::i>
                        - ExtentJMinus * GT_META_CALL(meta::at_c, (Strides, 1)){} // TODO access with dim::j
                    })
                    .template set<sid::property::strides>(
                        Strides{integral_constant<int_t, 1>{}, // TODO support for default init {} in hymap
                            BlockSizeI{},
                            integral_constant<int_t, BlockSizeI{} * BlockSizeJ{}>{},
                            GT_META_CALL(meta::at_c, (Strides, 2))::value *n_blocks_i,
                            GT_META_CALL(meta::at_c, (Strides, 2))::value *n_blocks_i *n_blocks_j})
                    .template set<sid::property::ptr_diff, int_t>()
                    .template set<sid::property::strides_kind, Strides>()));

} // namespace gridtools
