// this file will be preprocessed to visualize a repository
#include <gridtools/interface/repository/repository.hpp>
#define MY_FIELDTYPES (IJKDataStore, (0, 1, 2))(IJDataStore, (0, 1))
#define MY_FIELDS (IJKDataStore, u)(IJKDataStore, v)(IJDataStore, crlat)
GT_MAKE_REPOSITORY(my_repository, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS
