#pragma once

/**
   @file
   @brief File containing a set of metafunctions that apply to the storage classes.
*/

#include "storage.hpp"

namespace gridtools{

/**
 * @brief The storage_holds_data_field struct
 * determines if the storage class is holding a data field type of storage
 */
template<typename T>
struct storage_holds_data_field : boost::mpl::bool_<(T::field_dimensions > 1)>{};


    /**@brief metafunction to extract the metadata from a storage

    */
    template<typename Storage>
    struct storage2metadata{
        typedef typename Storage::meta_data_t
        type;
    };
}
