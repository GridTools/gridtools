/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "../common/layout_map.hpp"
#include "common/definitions.hpp"
#include "data_store.hpp"

#ifdef GT_USE_GPU
#include "storage_traits_cuda.hpp"

#include "storage_cuda/data_view_helpers.hpp"
#endif

#include "storage_traits_host.hpp"
#include "storage_traits_mc.hpp"

#include "storage_host/data_view_helpers.hpp"
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
