/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#pragma once

#include <cassert>
#include <type_traits>
#include <utility>

#include "../../common/cuda_util.hpp"
#include "../../common/gt_assert.hpp"
#include "../common/storage_info_interface.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    /*
     * @brief The cuda storage info implementation.
     * @tparam Id unique ID that should be shared among all storage infos with the same dimensionality.
     * @tparam Layout information about the memory layout
     * @tparam Halo information about the halo sizes (by default no halo is set)
     * @tparam Alignment information about the alignment (cuda_storage_info is aligned to 32 by default)
     */
    template <uint_t Id,
        typename Layout,
        typename Halo = zero_halo<Layout::masked_length>,
        typename Alignment = alignment<32>>
    using cuda_storage_info = storage_info_interface<Id, Layout, Halo, Alignment>;

    namespace impl_ {
        template <class SI>
        auto make_storage_info_ptr_cache(SI const &src)
            GT_AUTO_RETURN(std::make_pair(src, cuda_util::make_clone(src).release()));
    }

    /*
     * @brief retrieve the device pointer. This information is needed when the storage information should be passed
     * to a kernel.
     * @return a storage info device pointer
     */
    template <uint_t Id, typename Layout, typename Halo, typename Alignment>
    storage_info_interface<Id, Layout, Halo, Alignment> *get_gpu_storage_info_ptr(
        storage_info_interface<Id, Layout, Halo, Alignment> const &src) {
        thread_local static auto cache = impl_::make_storage_info_ptr_cache(src);
        if (cache.first != src)
            cache = impl_::make_storage_info_ptr_cache(src);
        return cache.second;
    }
    /**
     * @}
     */
} // namespace gridtools
