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
        struct default_layout<platform::cuda> {
            using type = layout_map<3, 2, 1, 0>;
        };
        template <>
        struct default_layout<platform::x86> {
            using type = layout_map<0, 1, 2, 3>;
        };
        template <>
        struct default_layout<platform::mc> {
            using type = layout_map<0, 1, 2, 3>;
        };
    } // namespace _impl

    /**
       The backend is, as usual, declaring what the storage types are
     */
    template <class BackendId, class StrategyType>
    struct backend<BackendId, grid_type::icosahedral, StrategyType>
        : public backend_base<BackendId, grid_type::icosahedral, StrategyType> {
      public:
        typedef backend_base<BackendId, grid_type::icosahedral, StrategyType> base_t;

        using typename base_t::backend_traits_t;
        using typename base_t::strategy_traits_t;

        template <typename DimSelector>
        struct select_layout {
            using layout_map_t = typename _impl::default_layout<BackendId>::type;
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
