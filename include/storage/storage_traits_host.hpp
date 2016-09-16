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
#include "../common/defs.hpp"
#include "storage.hpp"
#include "base_storage.hpp"
#include "meta_storage.hpp"
#include "meta_storage_aligned.hpp"
#include "meta_storage_base.hpp"

namespace gridtools {
    template < enumtype::platform T >
    struct storage_traits_from_id;

    /**Traits struct, containing the types which are specific for the host backend*/
    template <>
    struct storage_traits_from_id< enumtype::Host > {

        /**
           @brief pointer type associated to the host backend
         */
        template < typename T >
        struct pointer {
            typedef wrap_pointer< T, true > type;
        };

        /**
           @brief storage type associated to the host backend
         */
        template < typename ValueType, typename MetaData, short_t FieldDim = 1 >
        struct select_storage {
            GRIDTOOLS_STATIC_ASSERT((is_meta_storage< MetaData >::value), "wrong type for the storage_info");
            typedef storage< base_storage< typename pointer< ValueType >::type, MetaData, FieldDim > > type;
        };

        struct default_alignment {
            typedef aligned< 0 > type;
        };

        /**
           @brief storage info type associated to the host backend
         */
        template < typename IndexType, typename Layout, bool Temp, typename Halo, typename Alignment >
        struct select_meta_storage {
            GRIDTOOLS_STATIC_ASSERT((is_layout_map< Layout >::value), "wrong type for the storage_info");
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::type::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(is_aligned< Alignment >::type::value, "wrong type");

            typedef meta_storage<
                meta_storage_aligned< meta_storage_base< IndexType::value, Layout, Temp >, Alignment, Halo > > type;
        };
    };
}
