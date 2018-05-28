#include <gtest/gtest.h>

#include "exported_repository.hpp"

#define MY_FIELDTYPES (IJKDataStore, (0, 1, 2))(IJDataStore, (0, 1))
#define MY_FIELDS (IJKDataStore, ijkfield)(IJDataStore, ijfield)
GRIDTOOLS_MAKE_REPOSITORY_BINDINGS(exported_repository, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS

namespace {
    exported_repository make_exported_repository_impl(int Ni, int Nj, int Nk) {
        return exported_repository(IJKStorageInfo(Ni, Nj, Nk), IJStorageInfo(Ni, Nj));
    }
    GT_EXPORT_BINDING_3(make_exported_repository, make_exported_repository_impl);
    bool verify_exported_repository_impl(exported_repository &repository) {
        auto ijk_field = repository.ijkfield();
        auto ijk_view = make_host_view(ijk_field);

        int i = 0;
        for (int z = 0; z < ijk_field.dim< 2 >(); ++z) {
            for (int y = 0; y < ijk_field.dim< 1 >(); ++y) {
                for (int x = 0; x < ijk_field.dim< 0 >(); ++x) {
                    EXPECT_EQ(ijk_view(x, y, z), i++);
                }
            }
        }

        auto ij_field = repository.ijfield();
        auto ij_view = make_host_view(ij_field);
        i = 0;
        for (int y = 0; y < ij_field.dim< 1 >(); ++y) {
            for (int x = 0; x < ij_field.dim< 0 >(); ++x) {
                EXPECT_EQ(ij_view(x, y), i++);
            }
        }
    }
    GT_EXPORT_BINDING_1(verify_exported_repository, verify_exported_repository_impl);
}
