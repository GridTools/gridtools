#pragma once

/**
   @file
   @brief File containing a set of metafunctions that apply to the storage classes.
*/

#include "storage.hpp"

namespace gridtools{

template<typename T>
struct storage_holds_data_field : boost::mpl::false_{};

template <typename First,  typename  ...  StorageExtended>
struct storage_holds_data_field<storage<data_field<First, StorageExtended ... > > > : boost::mpl::true_ {};

}
