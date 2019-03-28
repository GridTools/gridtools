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
#include "../../common/functional.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple.hpp"
#include <memory>

using gridtools::host::constant;

namespace gridtools {
    class simple_cuda_allocator {
      public:
        std::shared_ptr<void> allocate(size_t bytes) {
            char *ptr;
            cudaMalloc(&ptr, bytes); // TODO wrap in error checker
            return std::shared_ptr<void>(ptr, [](char *ptr) { cudaFree(ptr); });
        }
    };

    struct block_i;
    struct block_j;

    // - k should be last dimension, then strides_kind doesn't need to distinguish between storages of different k-size
    // - if max extent of all temporaries is used, we don't have to distinguish any temporaries anymore
    template <class Extents>
    struct tmp_cuda_strides_kind;

    template <class T, class Extents, class BlockSizes /*, uint_t NColors*/>
    struct tmp_storage_cuda {
        using strides_t = tuple<integral_constant<int_t, 1>,
            integral_constant<int_t, tuple_util::get<0>(BlockSizes{})>,
            integral_constant<int_t, tuple_util::get<0>(BlockSizes{}) * tuple_util::get<1>(BlockSizes{})>,
            int_t,
            int_t>;

        template <class Allocator = int>
        tmp_storage_cuda(array<int_t, 2> n_blocks, int_t k_size, Allocator && = 1)
            : m_strides{{},
                  {},
                  {},
                  tuple_util::get<0>(BlockSizes{}) * tuple_util::get<1>(BlockSizes{}) * n_blocks[0],
                  tuple_util::get<0>(BlockSizes{}) * tuple_util::get<1>(BlockSizes{}) * n_blocks[0] * n_blocks[1]},
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
