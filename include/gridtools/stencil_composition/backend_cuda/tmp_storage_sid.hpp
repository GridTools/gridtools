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
#include <memory>

namespace gridtools {
    namespace tmp_cuda {
        struct block_i;
        struct block_j;
    } // namespace tmp_cuda

    template <class T,
        int_t ComputeBlockSizeI,
        int_t ComputeBlockSizeJ,
        int_t ExtentIMinus,
        int_t ExtentIPlus,
        int_t ExtentJMinus,
        int_t ExtentJPlus,
        class BlockSizeI = integral_constant<int_t, ComputeBlockSizeI - ExtentIMinus + ExtentIPlus>,
        class BlockSizeJ = integral_constant<int_t, ComputeBlockSizeJ - ExtentJMinus + ExtentJPlus>
        /*, uint_t NColors*/ // TODO duplicate with color
        >
    struct tmp_storage_cuda {
        using strides_t = hymap::keys<dim::i, dim::j, tmp_cuda::block_i, tmp_cuda::block_j, dim::k>::values<
            integral_constant<int_t, 1>,
            BlockSizeI,
            integral_constant<int_t, BlockSizeI{} * BlockSizeJ{}>,
            int_t,
            int_t>;

        const strides_t m_strides;
        host_device::constant<T *> m_cuda_ptr;

        template <class Allocator>
        tmp_storage_cuda(int_t n_blocks_i, int_t n_blocks_j, int_t k_size, Allocator &&alloc)
            : m_strides{integral_constant<int_t, 1>{},
                  BlockSizeI{},
                  integral_constant<int_t, BlockSizeI{} * BlockSizeJ{}>{},
                  meta::at_c<strides_t, 2>::value * n_blocks_i,
                  meta::at_c<strides_t, 2>::value * n_blocks_i * n_blocks_j},
              m_cuda_ptr{alloc.template allocate<T>(BlockSizeI{} * BlockSizeJ{} * n_blocks_i * n_blocks_j * k_size)} {}

        friend host_device::constant<T *> sid_get_origin(tmp_storage_cuda const &t) {
            return {t.m_cuda_ptr() - ExtentIMinus * at_key<dim::i>(t.m_strides) -
                    ExtentJMinus * at_key<dim::j>(t.m_strides)};
        }
        friend strides_t sid_get_strides(tmp_storage_cuda const &t) { return t.m_strides; }
        friend int_t sid_get_ptr_diff(tmp_storage_cuda const &) { return {}; }
        friend strides_t sid_get_strides_kind(tmp_storage_cuda const &) { return {}; }
    };

    //    template <class Allcoator>
    //    <unspec> make_tmp_storage_cuda() {}

} // namespace gridtools
