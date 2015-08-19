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
struct storage_holds_data_field : boost::mpl::false_{};


#ifdef CXX11_ENABLED
template <typename First,  typename  ...  StorageExtended>
struct storage_holds_data_field<storage<data_field<First, StorageExtended ... > > > : boost::mpl::true_ {};
#endif


    /**@brief metafunction to extract the metadata from a storage

    */
    template<typename Storage>
    struct storage2metadata{
        typedef typename Storage::meta_data_t
        type;
    };
}
