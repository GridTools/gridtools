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
#include "stencil-composition/arg_metafunctions_fwd.hpp"

namespace gridtools {
    /**
     * @struct arg_hods_data_field_h
     * high order metafunction of arg_holds_data_field
     */
    template < typename Arg >
    struct arg_holds_data_field_h {
        typedef typename arg_holds_data_field< typename Arg::type >::type type;
    };

    // metafunction to access the storage type given the arg
    template < typename T >
    struct arg2storage {
        typedef typename T::storage_type type;
    };

    // metafunction to access the metadata type given the arg
    template < typename T >
    struct arg2metadata {
        typedef typename arg2storage< T >::type::storage_info_type type;
    };

    /** metafunction extracting the location type from the storage*/
    template < typename T >
    struct get_location_type {
        typedef typename T::storage_info_type::index_type type;
    };

} // namespace gridtools
