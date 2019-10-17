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

#include "../common/defs.hpp"
#include "../common/layout_map.hpp"
#include "../common/selector.hpp"
#include "./common/halo.hpp"
#include "./common/storage_traits_metafunctions.hpp"
#include "./storage_mc/mc_storage.hpp"

namespace gridtools {
    template <class Backend>
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
    struct storage_traits_from_id<backend::mc> {
        static constexpr uint_t default_alignment = 8;

        template <typename ValueType>
        using select_storage = mc_storage<ValueType>;

        template <uint_t Dims>
        using select_layout = typename impl::layout_swap_mc<typename get_layout<Dims, false>::type>::type;
    };
} // namespace gridtools
