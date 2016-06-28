#pragma once

namespace gridtools {
    template < typename T >
    struct is_arg;

    /**
     * @struct arg_hods_data_field
     * metafunction that determines if an arg type is holding the storage type of a data field
     */
    template < typename Arg >
    struct arg_holds_data_field;

    template < typename T >
    struct is_arg_storage_pair;
}
