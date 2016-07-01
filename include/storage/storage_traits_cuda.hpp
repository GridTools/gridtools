/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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

    /** @brief traits struct defining the storage and metastorage types which are specific to the CUDA backend*/
    template <>
    struct storage_traits_from_id< enumtype::Cuda > {

        template < typename T >
        struct pointer {
            typedef hybrid_pointer< T, true > type;
        };

        template < typename ValueType, typename MetaData, short_t SpaceDim = 1 >
        struct select_storage {
            GRIDTOOLS_STATIC_ASSERT(is_meta_storage< MetaData >::value, "wrong type for the storage_info");
            typedef storage< base_storage< typename pointer< ValueType >::type, MetaData, SpaceDim > > type;
        };

        struct default_alignment {
            typedef aligned< 32 > type;
        };

        /**
           @brief storage info type associated to the cuda backend
           the storage info type is meta_storage.
         */
        template < typename IndexType, typename Layout, bool Temp, typename Halo, typename Alignment >
        struct select_meta_storage {
            GRIDTOOLS_STATIC_ASSERT(is_aligned< Alignment >::type::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::type::value, "wrong type");
            // GRIDTOOLS_STATIC_ASSERT((is_layout<Layout>::value), "wrong type for the storage_info");
            typedef meta_storage<
                meta_storage_aligned< meta_storage_base< IndexType, Layout, Temp >, Alignment, Halo > > type;
        };
    };
}
