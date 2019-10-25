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

#include <cassert>
#include <type_traits>
#include <utility>

#include "../../common/cuda_util.hpp"
#include "../../common/gt_assert.hpp"
#include "../common/storage_info.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    namespace impl_ {
        /*
         * @brief Allocates cuda_storage_info on device. Note that the pointer is released from the unique_ptr and
         * memory is leaked here. This is currently needed as otherwise the device storage_infos of (temporary) fields
         * with same ID but different storage size might be freed (and overwritten) before access in a stencil run()
         * method.
         */
        template <class SI>
        auto make_storage_info_ptr_cache(SI const &src) {
            return std::make_pair(src, cuda_util::make_clone(src).release());
        }
    } // namespace impl_

    /*
     * @brief retrieve the device pointer. This information is needed when the storage information should be passed
     * to a kernel.
     * @return a storage info device pointer
     */
    template <uint_t Id, typename Layout, typename Halo, typename Alignment>
    storage_info<Id, Layout, Halo, Alignment> *get_gpu_storage_info_ptr(
        storage_info<Id, Layout, Halo, Alignment> const &src) {
        thread_local static auto cache = impl_::make_storage_info_ptr_cache(src);
        if (cache.first != src)
            cache = impl_::make_storage_info_ptr_cache(src);
        return cache.second;
    }
    /**
     * @}
     */
} // namespace gridtools
