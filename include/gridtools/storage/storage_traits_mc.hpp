/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "../common/defs.hpp"
#include "../common/layout_map.hpp"
#include "../common/selector.hpp"
#include "./common/halo.hpp"
#include "./common/storage_traits_metafunctions.hpp"
#include "./storage_mc/mc_storage.hpp"
#include "./storage_mc/mc_storage_info.hpp"

namespace gridtools {
    template <class BackendId>
    struct storage_traits_from_id;

    namespace impl {
        template <class LayoutMap>
        struct layout_swap_mc {
            using type = LayoutMap;
        };

        template <int Dim0, int Dim1, int Dim2, int... Dims>
        struct layout_swap_mc<layout_map<Dim0, Dim1, Dim2, Dims...>> {
            using type = layout_map<Dim0, Dim2, Dim1, Dims...>;
        };
    } // namespace impl

    /** @brief storage traits for the Mic backend*/
    template <>
    struct storage_traits_from_id<target::mc> {

        template <typename ValueType>
        struct select_storage {
            using type = mc_storage<ValueType>;
        };

        template <uint_t Id, uint_t Dims, typename Halo>
        struct select_storage_info {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a halo type.");
#ifdef GT_STRUCTURED_GRIDS
            using layout = typename impl::layout_swap_mc<typename get_layout<Dims, false>::type>::type;
#else
            using layout = typename get_layout<Dims, true>::type;
#endif
            using type = mc_storage_info<Id, layout, Halo>;
        };

        template <uint_t Id, typename Layout, typename Halo>
        struct select_custom_layout_storage_info {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a halo type.");
            GT_STATIC_ASSERT(is_layout_map<Layout>::value, "Given type is not a layout map type.");
            using type = mc_storage_info<Id, Layout, Halo>;
        };

        template <uint_t Id, typename Selector, typename Halo>
        struct select_special_storage_info {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a halo type.");
            GT_STATIC_ASSERT(is_selector<Selector>::value, "Given type is not a selector type.");
#ifdef GT_STRUCTURED_GRIDS
            using layout = typename impl::layout_swap_mc<typename get_layout<Selector::size, false>::type>::type;
#else
            using layout = typename get_layout<Selector::size, true>::type;
#endif
            using type = mc_storage_info<Id, typename get_special_layout<layout, Selector>::type, Halo>;
        };

        template <uint_t Id, uint_t Dims, typename Halo, typename Align>
        struct select_storage_info_align {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a halo type.");
#ifdef GT_STRUCTURED_GRIDS
            using layout = typename impl::layout_swap_mc<typename get_layout<Dims, false>::type>::type;
#else
            using layout = typename get_layout<Dims, true>::type;
#endif
            using type = storage_info_interface<Id, layout, Halo, Align>;
        };

        template <uint_t Id, typename Layout, typename Halo, typename Align>
        struct select_custom_layout_storage_info_align {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a halo type.");
            GT_STATIC_ASSERT(is_layout_map<Layout>::value, "Given type is not a layout map type.");
            using type = storage_info_interface<Id, Layout, Halo, Align>;
        };

        template <uint_t Id, typename Selector, typename Halo, typename Align>
        struct select_special_storage_info_align {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a halo type.");
            GT_STATIC_ASSERT(is_selector<Selector>::value, "Given type is not a selector type.");
#ifdef GT_STRUCTURED_GRIDS
            using layout = typename impl::layout_swap_mc<typename get_layout<Selector::size, false>::type>::type;
#else
            using layout = typename get_layout<Selector::size, true>::type;
#endif
            using type = storage_info_interface<Id, typename get_special_layout<layout, Selector>::type, Halo, Align>;
        };
    };
} // namespace gridtools
