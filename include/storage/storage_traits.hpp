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
#include "../gridtools.hpp"
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
