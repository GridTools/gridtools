/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

#include "exported_repository.hpp"

#define MY_FIELDTYPES (IJKDataStore, (0, 1, 2))(IJDataStore, (0, 1, 2))(JKDataStore, (0, 1, 2))
#define MY_FIELDS (IJKDataStore, ijkfield)(IJDataStore, ijfield)(JKDataStore, jkfield)
GT_MAKE_REPOSITORY(exported_repository, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS

#define MY_FIELDS (IJKDataStore, ijkfield)(IJDataStore, ijfield)(JKDataStore, jkfield)
GT_MAKE_REPOSITORY_BINDINGS(exported_repository, exported, prefix_, MY_FIELDS)
#undef MY_FIELDS

namespace {
    exported_repository make_exported_repository_impl(int Ni, int Nj, int Nk) {
        return exported_repository(IJKStorageInfo(Ni, Nj, Nk), IJStorageInfo(Ni, Nj, Nk), JKStorageInfo(Ni, Nj, Nk));
    }
    BINDGEN_EXPORT_BINDING_3(make_exported_repository, make_exported_repository_impl);
    void verify_exported_repository_impl(exported_repository &repository) {
        auto ijk_field = repository.ijkfield();
        auto ijk_view = make_host_view(ijk_field);

        auto lengths = ijk_field.lengths();
        int i = 0;
        for (int z = 0; z < lengths[2]; ++z) {
            for (int y = 0; y < lengths[1]; ++y) {
                for (int x = 0; x < lengths[0]; ++x) {
                    EXPECT_EQ(ijk_view(x, y, z), i++);
                }
            }
        }

        auto ij_field = repository.ijfield();
        auto ij_view = make_host_view(ij_field);

        lengths = ij_field.lengths();
        i = 0;
        for (int y = 0; y < lengths[1]; ++y) {
            for (int x = 0; x < lengths[0]; ++x) {
                EXPECT_EQ(ij_view(x, y, 0), i++);
            }
        }

        auto jk_field = repository.jkfield();
        auto jk_view = make_host_view(jk_field);
        lengths = jk_field.lengths();
        i = 0;
        for (int z = 0; z < lengths[2]; ++z) {
            for (int y = 0; y < lengths[1]; ++y) {
                EXPECT_EQ(jk_view(0, y, z), i++);
            }
        }
    }
    BINDGEN_EXPORT_BINDING_1(verify_exported_repository, verify_exported_repository_impl);
} // namespace
