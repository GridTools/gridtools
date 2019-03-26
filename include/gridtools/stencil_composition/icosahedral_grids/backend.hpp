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

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/shorten.hpp"
#include "../../common/layout_map_metafunctions.hpp"
#include "../backend_base.hpp"
#include "../backend_fwd.hpp"

namespace gridtools {

    namespace _impl {
        template <class>
        struct default_layout;
        template <>
        struct default_layout<target::cuda> {
            using type = layout_map<3, 2, 1, 0>;
        };
        template <>
        struct default_layout<target::x86> {
            using type = layout_map<0, 1, 2, 3>;
        };
        template <>
        struct default_layout<target::naive> {
            using type = layout_map<0, 1, 2, 3>;
        };
    } // namespace _impl

    /**
       The backend is, as usual, declaring what the storage types are
     */
    template <class BackendTarget>
    struct backend : public backend_base<BackendTarget> {
      public:
        typedef backend_base<BackendTarget> base_t;

        using typename base_t::backend_traits_t;

        template <typename DimSelector>
        struct select_layout {
            using layout_map_t = typename _impl::default_layout<BackendTarget>::type;
            using dim_selector_4d_t = typename shorten<bool, DimSelector, 4>::type;
            using filtered_layout = typename filter_layout<layout_map_t, dim_selector_4d_t>::type;

            using type = typename boost::mpl::eval_if_c<(DimSelector::size > 4),
                extend_layout_map<filtered_layout, DimSelector::size - 4>,
                boost::mpl::identity<filtered_layout>>::type;
        };

        template <unsigned Index, typename LayoutMap, typename Halo>
        using storage_info_t =
            typename base_t::storage_traits_t::template custom_layout_storage_info_t<Index, LayoutMap, Halo>;

        template <typename ValueType, typename StorageInfo>
        using data_store_t = typename base_t::storage_traits_t::template data_store_t<ValueType, StorageInfo>;
    };
} // namespace gridtools
