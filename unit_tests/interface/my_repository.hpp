#include <unordered_map>
#include "storage/storage-facility.hpp"
#include "boost/variant.hpp"

using namespace gridtools;

using IJKStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 0, 3 >;
using IJKDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJKStorageInfo >;
using IJStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 1, 2 >;
using IJDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJStorageInfo >;

#define GT_REPOSITORY_NAME my_repository
#define GT_REPOSITORY_INC "../../../unit_tests/interface/my_repository.inc"
#include "interface/repository/repository.hpp"

#define GT_REPOSITORY_NAME my_repository2
#define GT_REPOSITORY_INC "../../../unit_tests/interface/my_repository2.inc"
#include "interface/repository/repository.hpp"
