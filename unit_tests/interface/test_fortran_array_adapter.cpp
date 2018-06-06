/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#include <gtest/gtest.h>

#include <gridtools/interface/fortran_array_adapter.hpp>
#include <gridtools/c_bindings/fortran_array_view.hpp>
#include <gridtools/storage/storage-facility.hpp>

using IJKStorageInfo = typename gridtools::storage_traits< gridtools::enumtype::Host >::storage_info_t< 0, 3 >;
using IJKDataStore =
    typename gridtools::storage_traits< gridtools::enumtype::Host >::data_store_t< gridtools::float_type,
        IJKStorageInfo >;

TEST(FortranArrayAdapter, TransformAdapterIntoDataStore) {
    constexpr size_t x_size = 6;
    constexpr size_t y_size = 5;
    constexpr size_t z_size = 4;
    gridtools::float_type fortran_array[z_size][y_size][x_size];

    gt_fortran_array_descriptor descriptor;
    descriptor.rank = 3;
    descriptor.dims[0] = x_size;
    descriptor.dims[1] = y_size;
    descriptor.dims[2] = z_size;
    descriptor.type = std::is_same< gridtools::float_type, float >::value ? gt_fk_Float : gt_fk_Double;
    descriptor.data = fortran_array;

    gridtools::fortran_array_adapter< IJKDataStore > fortran_array_adapter{descriptor};
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
    gridtools::float_type fortran_array[z_size][y_size][x_size];

    gt_fortran_array_descriptor descriptor;
    descriptor.rank = 3;
    descriptor.dims[0] = x_size;
    descriptor.dims[1] = y_size;
    descriptor.dims[2] = z_size;
    descriptor.type = std::is_same< gridtools::float_type, float >::value ? gt_fk_Float : gt_fk_Double;
    descriptor.data = fortran_array;

    gridtools::fortran_array_adapter< IJKDataStore > fortran_array_adapter{descriptor};
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
