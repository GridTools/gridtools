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

#include "../common/layout_map.hpp"
#include "common/definitions.hpp"
#include "common/halo.hpp"
#include "data_store.hpp"
#include "data_view.hpp"

#ifdef GT_USE_GPU
#include "storage_traits_cuda.hpp"

#endif

#include "storage_traits_mc.hpp"
#include "storage_traits_naive.hpp"
#include "storage_traits_x86.hpp"

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
    template <class Backend>
    class storage_traits {
        template <typename ValueType>
        using storage_type = typename storage_traits_from_id<Backend>::template select_storage<ValueType>;

        template <uint_t Dims>
        using layout_type = typename storage_traits_from_id<Backend>::template select_layout<Dims>;

        using default_alignment_t = alignment<storage_traits_from_id<Backend>::default_alignment>;

      public:
        template <uint_t Id, uint_t Dims, typename Halo = zero_halo<Dims>, typename Align = default_alignment_t>
        using storage_info_t = storage_info<Id, layout_type<Dims>, Halo, Align>;

        template <uint_t Id,
            typename Layout,
            typename Halo = zero_halo<Layout::masked_length>,
            typename Align = default_alignment_t>
        using custom_layout_storage_info_t = storage_info<Id, Layout, Halo, Align>;

        template <uint_t Id,
            typename Selector,
            typename Halo = zero_halo<Selector::size()>,
            typename Align = default_alignment_t>
        using special_storage_info_t =
            storage_info<Id, typename get_special_layout<layout_type<Selector::size()>, Selector>::type, Halo, Align>;

        template <typename ValueType, typename StorageInfo>
        using data_store_t = data_store<storage_type<ValueType>, StorageInfo>;
    };

    /**
     * @}
     */
} // namespace gridtools
