
#include <unordered_map>
#include "storage/storage-facility.hpp"
#include "boost/variant.hpp"

using namespace gridtools;

using IJKStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 0, 3 >;
using IJKDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJKStorageInfo >;
using IJStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 1, 2 >;
using IJDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJStorageInfo >;

#define GT_REPOSITORY_FIELDTYPES (IJKDataStore)(IJDataStore)
#define GT_REPOSITORY_FIELDS (IJKDataStore, u)(IJKDataStore, w)(IJKDataStore, v)(IJDataStore, crlat)
#define GT_REPOSITORY_NAME my_repository4
#include "interface/repository/repository.hpp"

class my_useful_repo : public my_repository4 {
    using my_repository4::my_repository4;
    void super_useful_function() { std::cout << "bla" << std::endl; }
};
