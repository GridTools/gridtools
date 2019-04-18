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

#include "../common/gt_assert.hpp"
#include "../common/selector.hpp"
#include "./common/definitions.hpp"
#include "./common/storage_traits_metafunctions.hpp"
#include "./storage_cuda/cuda_storage.hpp"
#include "./storage_cuda/cuda_storage_info.hpp"
#include "./storage_cuda/data_view_helpers.hpp"

namespace gridtools {
    /** \ingroup storage
     * @{
     */

    template <class Backend>
    struct storage_traits_from_id;

    /** @brief storage traits for the CUDA backend*/
    template <>
    struct storage_traits_from_id<backend::cuda> {

        template <typename ValueType>
        struct select_storage {
            typedef cuda_storage<ValueType> type;
        };

        template <uint_t Id, uint_t Dims, typename Halo>
        struct select_storage_info {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a Halo type.");
            typedef typename get_layout<Dims, false>::type layout;
            typedef cuda_storage_info<Id, layout, Halo> type;
        };

        template <uint_t Id, typename Layout, typename Halo>
        struct select_custom_layout_storage_info {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a halo type.");
            GT_STATIC_ASSERT(is_layout_map<Layout>::value, "Given type is not a layout map type.");
            typedef cuda_storage_info<Id, Layout, Halo> type;
        };

        template <uint_t Id, typename Selector, typename Halo>
        struct select_special_storage_info {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a Halo type.");
            GT_STATIC_ASSERT(is_selector<Selector>::value, "Given type is not a selector type.");
            typedef typename get_layout<Selector::size(), false>::type layout;
            typedef cuda_storage_info<Id, typename get_special_layout<layout, Selector>::type, Halo> type;
        };

        template <uint_t Id, uint_t Dims, typename Halo, typename Align>
        struct select_storage_info_align {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a Halo type.");
            typedef typename get_layout<Dims, false>::type layout;
            typedef storage_info<Id, layout, Halo, Align> type;
        };

        template <uint_t Id, typename Layout, typename Halo, typename Align>
        struct select_custom_layout_storage_info_align {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a halo type.");
            GT_STATIC_ASSERT(is_layout_map<Layout>::value, "Given type is not a layout map type.");
            typedef storage_info<Id, Layout, Halo, Align> type;
        };

        template <uint_t Id, typename Selector, typename Halo, typename Align>
        struct select_special_storage_info_align {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a Halo type.");
            GT_STATIC_ASSERT(is_selector<Selector>::value, "Given type is not a selector type.");
            typedef typename get_layout<Selector::size(), false>::type layout;
            typedef storage_info<Id, typename get_special_layout<layout, Selector>::type, Halo, Align> type;
        };
    };

    /**
     * @}
     */
} // namespace gridtools
