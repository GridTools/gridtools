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

#include "../common/layout_map.hpp"
#include "common/definitions.hpp"
#include "data_store.hpp"
#include "data_store_field.hpp"

#ifdef _USE_GPU_
#include "storage_traits_cuda.hpp"

#include "storage_cuda/data_field_view_helpers.hpp"
#include "storage_cuda/data_view_helpers.hpp"
#endif

#include "storage_traits_host.hpp"
#include "storage_traits_mc.hpp"

#include "storage_host/data_field_view_helpers.hpp"
#include "storage_host/data_view_helpers.hpp"
#include "storage_mc/data_field_view_helpers.hpp"
#include "storage_mc/data_view_helpers.hpp"

/**
 * \defgroup storage Storage
 */

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    /**
     * @brief storage traits used to retrieve the correct storage_info, data_store, and data_store_field types.
     * Additionally to the default types, specialized and custom storage_info types can be retrieved
     * @tparam T used target (e.g., Cuda or Host)
     */
    template <class BackendId>
    struct storage_traits : gridtools::storage_traits_from_id<BackendId> {
      private:
        template <typename ValueType>
        using storage_t =
            typename gridtools::storage_traits_from_id<BackendId>::template select_storage<ValueType>::type;

      public:
        template <uint_t Id, uint_t Dims, typename Halo = zero_halo<Dims>>
        using storage_info_t =
            typename gridtools::storage_traits_from_id<BackendId>::template select_storage_info<Id, Dims, Halo>::type;

        template <uint_t Id, typename LayoutMap, typename Halo = zero_halo<LayoutMap::masked_length>>
        using custom_layout_storage_info_t = typename gridtools::storage_traits_from_id<
            BackendId>::template select_custom_layout_storage_info<Id, LayoutMap, Halo>::type;

        template <uint_t Id, typename Selector, typename Halo = zero_halo<Selector::size>>
        using special_storage_info_t = typename gridtools::storage_traits_from_id<
            BackendId>::template select_special_storage_info<Id, Selector, Halo>::type;

        template <typename ValueType, typename StorageInfo>
        using data_store_t = data_store<storage_t<ValueType>, StorageInfo>;

        template <typename ValueType, typename StorageInfo, uint_t... N>
        using data_store_field_t = data_store_field<data_store_t<ValueType, StorageInfo>, N...>;

        template <uint_t Id, uint_t Dims, typename Halo, typename Align>
        using storage_info_align_t = typename gridtools::storage_traits_from_id<
            BackendId>::template select_storage_info_align<Id, Dims, Halo, Align>::type;

        template <uint_t Id, typename LayoutMap, typename Halo, typename Align>
        using custom_layout_storage_info_align_t = typename gridtools::storage_traits_from_id<
            BackendId>::template select_custom_layout_storage_info_align<Id, LayoutMap, Halo, Align>::type;

        template <uint_t Id, typename Selector, typename Halo, typename Align>
        using special_storage_info_align_t = typename gridtools::storage_traits_from_id<
            BackendId>::template select_special_storage_info_align<Id, Selector, Halo, Align>::type;
    };

    /**
     * @}
     */
} // namespace gridtools
