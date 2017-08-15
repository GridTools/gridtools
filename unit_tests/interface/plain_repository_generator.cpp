#include "interface/repository/repository.hpp"
#define MY_FIELDTYPES (IJKDataStore, (0, 1, 2))(IJDataStore, (0, 1))
#define MY_FIELDS (IJKDataStore, u)(IJKDataStore, v)(IJDataStore, crlat)
GRIDTOOLS_MAKE_REPOSITORY(my_repository, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS
