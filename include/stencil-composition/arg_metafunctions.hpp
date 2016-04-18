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

} // namespace gridtools
