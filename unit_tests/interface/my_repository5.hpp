
#include "storage/storage-facility.hpp"
#include "interface/repository/repository.hpp"

using namespace gridtools;

using IJKStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 0, 3 >;
using IJKDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJKStorageInfo >;
using IJStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 1, 2 >;
using IJDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJStorageInfo >;

#define MY_FIELDTYPES (IJKDataStore)(IJDataStore)
#define MY_FIELDS (IJKDataStore, u)(IJDataStore, crlat)
GT_MAKE_REPOSITORY(my_repository5, MY_FIELDTYPES, MY_FIELDS)

class my_useful_repo : public my_repository5 {
    using my_repository5::my_repository5;
    void super_useful_function() { std::cout << "bla" << std::endl; }
};
