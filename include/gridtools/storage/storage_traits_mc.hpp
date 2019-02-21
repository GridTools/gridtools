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
            using type = storage_info<Id, layout, Halo, Align>;
        };

        template <uint_t Id, typename Layout, typename Halo, typename Align>
        struct select_custom_layout_storage_info_align {
            GT_STATIC_ASSERT(is_halo<Halo>::value, "Given type is not a halo type.");
            GT_STATIC_ASSERT(is_layout_map<Layout>::value, "Given type is not a layout map type.");
            using type = storage_info<Id, Layout, Halo, Align>;
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
            using type = storage_info<Id, typename get_special_layout<layout, Selector>::type, Halo, Align>;
        };
    };
} // namespace gridtools
