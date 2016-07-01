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
#include "../common/layout_map.hpp"
#include "halo.hpp"

#include "storage_traits_cuda.hpp"
#include "storage_traits_host.hpp"

namespace gridtools {

    template < enumtype::platform T >
    struct storage_traits {

        typedef gridtools::storage_traits_from_id< T > storage_traits_aux;

        template < typename ValueType, typename MetaData >
        struct get_temporary_storage_type_aux {
            GRIDTOOLS_STATIC_ASSERT(is_meta_storage< MetaData >::value, "wrong type for the storage_info");
            // convert the meta data type into a temporary meta data type
            typedef typename storage_traits_aux::template select_meta_storage< typename MetaData::index_type,
                typename MetaData::layout,
                true,
                typename MetaData::halo_t,
                typename MetaData::alignment_t >::type meta_data_tmp;
            // create new storage that takes temporary meta data as storage_info
            typedef typename storage_traits_aux::template select_storage< ValueType, meta_data_tmp >::type storage_type;
            // create final temporary storage type (no_storage_type_yet)
            typedef no_storage_type_yet< storage_type > type;
        };

// convenience structs that unwrap the inner storage and meta_storage type
#ifdef CXX11_ENABLED
        template < typename ValueType, typename MetaData >
        using storage_type = typename storage_traits_aux::template select_storage< ValueType, MetaData >::type;

        template < ushort_t Index,
            typename Layout,
            typename Halo = halo< 0, 0, 0 >,
            typename Alignment = typename storage_traits_aux::default_alignment::type >
        using meta_storage_type = typename storage_traits_aux::
            template select_meta_storage< static_uint< Index >, Layout, false, Halo, Alignment >::type;

        template < typename ValueType, typename MetaData >
        using temporary_storage_type = typename get_temporary_storage_type_aux< ValueType, MetaData >::type;

#else // CXX11_ENABLED
        template < typename ValueType, typename MetaData >
        struct storage_type {
            typedef typename storage_traits_aux::template select_storage< ValueType, MetaData >::type type;
        };

        template < ushort_t Index,
            typename Layout,
            typename Halo = halo< 0, 0, 0 >,
            typename Alignment = typename storage_traits_aux::default_alignment::type >
        struct meta_storage_type {
            typedef typename storage_traits_aux::
                template select_meta_storage< static_uint< Index >, Layout, false, Halo, Alignment >::type type;
        };

        template < typename ValueType, typename MetaData >
        struct temporary_storage_type : get_temporary_storage_type_aux< ValueType, MetaData > {};
#endif
    };
};
