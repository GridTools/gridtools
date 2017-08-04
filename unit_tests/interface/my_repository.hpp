
#include "storage/storage-facility.hpp"
#include "interface/repository/repository.hpp"

using namespace gridtools;

using IJKStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 0, 3 >;
using IJKDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJKStorageInfo >;
using IJStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 1, 2 >;
using IJDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJStorageInfo >;

#define MY_FIELDTYPES (IJKDataStore)(IJDataStore)
#define MY_FIELDS (IJKDataStore, u)(IJDataStore, crlat)
GT_MAKE_REPOSITORY(my_repository, MY_FIELDTYPES, MY_FIELDS)

class my_extended_repo : public my_repository {
    using my_repository::my_repository;
    void some_extra_function() {}
};
