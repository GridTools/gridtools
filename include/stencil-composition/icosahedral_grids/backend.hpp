/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include "storage/storage.hpp"
#include "storage/meta_storage.hpp"
#include "location_type.hpp"
#include "stencil-composition/backend_base.hpp"
#include "storage/wrap_pointer.hpp"
#include "icosahedral_grid_traits.hpp"
#include "common/selector.hpp"

namespace gridtools {

    /**
       The backend is, as usual, declaring what the storage types are
     */
    template < enumtype::platform BackendId, enumtype::strategy StrategyType >
    struct backend< BackendId, enumtype::icosahedral, StrategyType >
        : public backend_base< BackendId, enumtype::icosahedral, StrategyType > {
      public:
        typedef backend_base< BackendId, enumtype::icosahedral, StrategyType > base_t;

        using typename base_t::backend_traits_t;
        using typename base_t::strategy_traits_t;
        using layout_map_t = typename icgrid::grid_traits_arch< base_t::s_backend_id >::layout_map_t;

        template < typename DimSelector >
        using select_layout = typename filter_layout< layout_map_t, DimSelector >::type;

        template < uint_t Index,
            typename LayoutMap >
        using storage_info_t = typename base_t::template storage_info< Index, LayoutMap >;

        template< typename ValueType,
            typename StorageInfo>
        using storage_t = typename base_t::template storage_type< ValueType, StorageInfo>::type;

    };
} // namespace gridtools
