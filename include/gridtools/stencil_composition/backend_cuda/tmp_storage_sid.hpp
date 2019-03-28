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

#include "../../common/array.hpp"
#include "../../common/cuda_util.hpp"
#include "../../common/functional.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple.hpp"
#include "../../meta/at.hpp"
#include "../dim.hpp"
#include <memory>

using gridtools::host::constant;

namespace gridtools {
    class simple_cuda_allocator {
      public:
        std::shared_ptr<void> allocate(size_t bytes) {
            char *ptr;
            GT_CUDA_CHECK(cudaMalloc(&ptr, bytes));
            return std::shared_ptr<void>(ptr, [](char *ptr) { GT_CUDA_CHECK(cudaFree(ptr)); });
        }
    };

    struct block_i;
    struct block_j;

    // - k is the last dimension, then strides_kind doesn't need to distinguish between storages of different k-size
    // - If max extent of all temporaries is used (instead of per temporary extent),
    //   the strides_kind can be the same for all temporaries.
    template <class Extents>
    struct tmp_cuda_strides_kind;

    template <class T,
        class Extents,                                // TODO how to pass extents?
        class ComputeBlockSizes /*, uint_t NColors*/, // TODO separate storage for icosahedral or implement here?
        class BlockSizeI = integral_constant<int_t,
            meta::at_c<ComputeBlockSizes, 0>::value + meta::at_c<meta::at_c<Extents, 0>, 0>::value +
                meta::at_c<meta::at_c<Extents, 0>, 1>::value>, // TODO make readable // TODO uglify
        class BlockSizeJ = integral_constant<int_t,
            meta::at_c<ComputeBlockSizes, 1>::value + meta::at_c<meta::at_c<Extents, 1>, 0>::value +
                meta::at_c<meta::at_c<Extents, 1>, 1>::value>>
    struct tmp_storage_cuda {
        using strides_t = hymap::keys<dim::i, dim::j, block_i, block_j, dim::k>::values<integral_constant<int_t, 1>,
            BlockSizeI,
            integral_constant<int_t, BlockSizeI{} * BlockSizeJ{}>,
            int_t,
            int_t>;

        template <class Allocator = int>
        tmp_storage_cuda(array<int_t, 2> n_blocks, int_t k_size, Allocator && = 1)
            : m_strides{{},
                  {},
                  {},
                  meta::at_c<strides_t, 2>::value * n_blocks[0],
                  meta::at_c<strides_t, 2>::value * n_blocks[0] * n_blocks[1]},
              m_cuda_ptr{new T()} {}

        const strides_t m_strides;
        std::shared_ptr<void> m_cuda_ptr;

        friend host_device::constant<T *> sid_get_origin(tmp_storage_cuda const &t) {
            return {static_cast<T *>(t.m_cuda_ptr.get())};
        }
        friend strides_t sid_get_strides(tmp_storage_cuda const &t) { return t.m_strides; }
        friend int_t sid_get_ptr_diff(tmp_storage_cuda const &);
        friend tmp_cuda_strides_kind<Extents> sid_get_strides_kind(tmp_storage_cuda const &);
    };

} // namespace gridtools
