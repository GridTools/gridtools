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

#include <cpp_bindgen/fortran_array_view.hpp>
#include <gridtools/interface/fortran_array_adapter.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gridtools/tools/backend_select.hpp>

using IJKStorageInfo = typename gridtools::storage_traits<gridtools::backend::x86>::storage_info_t<0, 3>;
using IJKDataStore =
    typename gridtools::storage_traits<gridtools::backend::x86>::data_store_t<float_type, IJKStorageInfo>;

TEST(FortranArrayAdapter, TransformAdapterIntoDataStore) {
    constexpr size_t x_size = 6;
    constexpr size_t y_size = 5;
    constexpr size_t z_size = 4;
    float_type fortran_array[z_size][y_size][x_size];

    bindgen_fortran_array_descriptor descriptor;
    descriptor.rank = 3;
    descriptor.dims[0] = x_size;
    descriptor.dims[1] = y_size;
    descriptor.dims[2] = z_size;
    descriptor.type = std::is_same<float_type, float>::value ? bindgen_fk_Float : bindgen_fk_Double;
    descriptor.data = fortran_array;
    descriptor.is_acc_present = false;

    gridtools::fortran_array_adapter<IJKDataStore> fortran_array_adapter{descriptor};
    IJKDataStore data_store{IJKStorageInfo{x_size, y_size, z_size}};
    auto data_store_view = make_host_view(data_store);

    int i = 0;
    for (size_t z = 0; z < z_size; ++z)
        for (size_t y = 0; y < y_size; ++y)
            for (size_t x = 0; x < x_size; ++x, ++i)
                fortran_array[z][y][x] = i;

    // transform adapter into data_store
    transform(data_store, fortran_array_adapter);

    i = 0;
    for (size_t z = 0; z < z_size; ++z)
        for (size_t y = 0; y < y_size; ++y)
            for (size_t x = 0; x < x_size; ++x, ++i)
                EXPECT_EQ(data_store_view(x, y, z), i);
}

TEST(FortranArrayAdapter, TransformDataStoreIntoAdapter) {
    constexpr size_t x_size = 6;
    constexpr size_t y_size = 5;
    constexpr size_t z_size = 4;
    float_type fortran_array[z_size][y_size][x_size];

    bindgen_fortran_array_descriptor descriptor;
    descriptor.rank = 3;
    descriptor.dims[0] = x_size;
    descriptor.dims[1] = y_size;
    descriptor.dims[2] = z_size;
    descriptor.type = std::is_same<float_type, float>::value ? bindgen_fk_Float : bindgen_fk_Double;
    descriptor.data = fortran_array;
    descriptor.is_acc_present = false;

    gridtools::fortran_array_adapter<IJKDataStore> fortran_array_adapter{descriptor};
    IJKDataStore data_store{IJKStorageInfo{x_size, y_size, z_size}};
    auto data_store_view = make_host_view(data_store);

    int i = 0;
    for (size_t z = 0; z < z_size; ++z)
        for (size_t y = 0; y < y_size; ++y)
            for (size_t x = 0; x < x_size; ++x, ++i)
                data_store_view(x, y, z) = i;

    // transform data_store into adapter
    transform(fortran_array_adapter, data_store);

    i = 0;
    for (size_t z = 0; z < z_size; ++z)
        for (size_t y = 0; y < y_size; ++y)
            for (size_t x = 0; x < x_size; ++x, ++i)
                EXPECT_EQ(fortran_array[z][y][x], i);
}
