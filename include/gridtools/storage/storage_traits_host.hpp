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

#include "common/selector.hpp"
#include "common/gt_assert.hpp"
#include "storage/common/definitions.hpp"
#include "storage/common/storage_traits_metafunctions.hpp"
#include "storage/storage_host/data_field_view_helpers.hpp"
#include "storage/storage_host/data_view_helpers.hpp"
#include "storage/storage_host/host_storage.hpp"
#include "storage/storage_host/host_storage_info.hpp"

namespace gridtools {
    /** \ingroup storage
     * @{
     */

    template < enumtype::platform T >
    struct storage_traits_from_id;

    /** @brief storage traits for the Host backend*/
    template <>
    struct storage_traits_from_id< enumtype::Host > {

        template < typename ValueType >
        struct select_storage {
            typedef host_storage< ValueType > type;
        };

        template < uint_t Id, uint_t Dims, typename Halo >
        struct select_storage_info {
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::value, "Given type is not a halo type.");
            typedef typename get_layout< Dims, true >::type layout;
            typedef host_storage_info< Id, layout, Halo > type;
        };

        template < uint_t Id, typename Layout, typename Halo >
        struct select_custom_layout_storage_info {
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::value, "Given type is not a halo type.");
            GRIDTOOLS_STATIC_ASSERT(is_layout_map< Layout >::value, "Given type is not a layout map type.");
            typedef host_storage_info< Id, Layout, Halo > type;
        };

        template < uint_t Id, typename Selector, typename Halo >
        struct select_special_storage_info {
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::value, "Given type is not a halo type.");
            GRIDTOOLS_STATIC_ASSERT(is_selector< Selector >::value, "Given type is not a selector type.");
            typedef typename get_layout< Selector::size, true >::type layout;
            typedef host_storage_info< Id, typename get_special_layout< layout, Selector >::type, Halo > type;
        };

        template < uint_t Id, uint_t Dims, typename Halo, typename Align >
        struct select_storage_info_align {
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::value, "Given type is not a halo type.");
            typedef typename get_layout< Dims, true >::type layout;
            typedef host_storage_info< Id, layout, Halo, Align > type;
        };

        template < uint_t Id, typename Layout, typename Halo, typename Align >
        struct select_custom_layout_storage_info_align {
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::value, "Given type is not a halo type.");
            GRIDTOOLS_STATIC_ASSERT(is_layout_map< Layout >::value, "Given type is not a layout map type.");
            typedef host_storage_info< Id, Layout, Halo, Align > type;
        };

        template < uint_t Id, typename Selector, typename Halo, typename Align >
        struct select_special_storage_info_align {
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::value, "Given type is not a halo type.");
            GRIDTOOLS_STATIC_ASSERT(is_selector< Selector >::value, "Given type is not a selector type.");
            typedef typename get_layout< Selector::size, true >::type layout;
            typedef host_storage_info< Id, typename get_special_layout< layout, Selector >::type, Halo, Align > type;
        };
    };

    /**
     * @}
     */
}
