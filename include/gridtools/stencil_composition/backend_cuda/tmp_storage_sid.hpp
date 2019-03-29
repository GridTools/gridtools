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
#include "../../common/cuda_allocator.hpp"
#include "../../common/cuda_util.hpp"
#include "../../common/functional.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple.hpp"
#include "../../meta/at.hpp"
#include "../dim.hpp"
#include <memory>

namespace gridtools {
    namespace tmp_storage_sid_impl_ {
        template <class Extents, class ComputeBlockSizes>
        using compute_bock_size_i = integral_constant<int_t,
            meta::at_c<ComputeBlockSizes, 0>::value + meta::at_c<meta::at_c<Extents, 0>, 0>::value +
                meta::at_c<meta::at_c<Extents, 0>, 1>::value>; // TODO uglify

        template <class Extents, class ComputeBlockSizes>
        using compute_bock_size_j = integral_constant<int_t,
            meta::at_c<ComputeBlockSizes, 1>::value + meta::at_c<meta::at_c<Extents, 1>, 0>::value +
                meta::at_c<meta::at_c<Extents, 1>, 1>::value>; // TODO uglify
    }                                                          // namespace tmp_storage_sid_impl_

    namespace tmp_cuda {
        struct block_i;
        struct block_j;
    } // namespace tmp_cuda

    // - k is the last dimension, then strides_kind doesn't need to distinguish between storages of different k-size
    // - If max extent of all temporaries is used (instead of per temporary extent),
    //   the strides_kind can be the same for all temporaries.
    template <class Extents>
    struct tmp_cuda_strides_kind;

    template <class T,
        class Extents, // TODO how to pass extents?
        class ComputeBlockSizes
        /*, uint_t NColors*/ // TODO separate storage for icosahedral or implement here?
        >
    struct tmp_storage_cuda {
        using strides_t = hymap::keys<dim::i, dim::j, tmp_cuda::block_i, tmp_cuda::block_j, dim::k>::values<
            integral_constant<int_t, 1>,
            tmp_storage_sid_impl_::compute_bock_size_i<Extents, ComputeBlockSizes>,
            integral_constant<int_t,
                tmp_storage_sid_impl_::compute_bock_size_i<Extents, ComputeBlockSizes>{} *
                    tmp_storage_sid_impl_::compute_bock_size_j<Extents, ComputeBlockSizes>{}>,
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
        friend int_t sid_get_ptr_diff(tmp_storage_cuda const &) { return {}; };
        friend tmp_cuda_strides_kind<Extents> sid_get_strides_kind(tmp_storage_cuda const &) { return {}; };
    };

} // namespace gridtools
