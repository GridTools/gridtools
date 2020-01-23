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

#include <cpp_bindgen/export.hpp>

#include <gridtools/interface/repository.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/x86.hpp>
#include <gridtools/tools/backend_select.hpp>

namespace {
    using namespace gridtools;

    const auto builder = storage::builder<storage::x86>.type<float_type>();

    auto ijk_builder(int i, int j, int k) { return builder.dimensions(i, j, k); }
    auto ij_builder(int i, int j, int k) { return builder.selector<1, 1, 0>().dimensions(i, j, k); }
    auto jk_builder(int i, int j, int k) { return builder.selector<0, 1, 1>().dimensions(i, j, k); }

    GT_DEFINE_REPOSITORY(
        repo, (ijk, ijk_builder)(ij, ij_builder)(jk, jk_builder), (ijk, ijkfield)(ij, ijfield)(jk, jkfield));

    GT_DEFINE_REPOSITORY_BINDINGS(repo, prefix_set_exported_, ijkfield, ijfield, jkfield);

    repo make_exported_repository_impl(int Ni, int Nj, int Nk) { return {Ni, Nj, Nk}; }
    BINDGEN_EXPORT_BINDING_3(make_exported_repository, make_exported_repository_impl);

    void verify_exported_repository_impl(repo const &repo) {
        {
            auto &&view = repo.ijkfield->host_view();
            auto &&lengths = repo.ijkfield->lengths();
            int i = 0;
            for (int z = 0; z < lengths[2]; ++z)
                for (int y = 0; y < lengths[1]; ++y)
                    for (int x = 0; x < lengths[0]; ++x)
                        EXPECT_EQ(view(x, y, z), i++);
        }
        {
            auto &&view = repo.ijfield->host_view();
            auto &&lengths = repo.ijfield->lengths();
            int i = 0;
            for (int y = 0; y < lengths[1]; ++y)
                for (int x = 0; x < lengths[0]; ++x)
                    EXPECT_EQ(view(x, y, 0), i++);
        }
        {
            auto &&view = repo.jkfield->host_view();
            auto &&lengths = repo.jkfield->lengths();
            int i = 0;
            for (int z = 0; z < lengths[2]; ++z)
                for (int y = 0; y < lengths[1]; ++y)
                    EXPECT_EQ(view(0, y, z), i++);
        }
    }
    BINDGEN_EXPORT_BINDING_1(verify_exported_repository, verify_exported_repository_impl);
} // namespace
