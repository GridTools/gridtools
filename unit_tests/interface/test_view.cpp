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

#include "interface/View.hpp"
#include "c_bindings/fortran_array_view.hpp"
#include "storage/storage-facility.hpp"

using IJKStorageInfo = typename gridtools::storage_traits< gridtools::enumtype::Host >::storage_info_t< 0, 3 >;
using IJKDataStore =
    typename gridtools::storage_traits< gridtools::enumtype::Host >::data_store_t< gridtools::float_type,
        IJKStorageInfo >;

TEST(View, TransformViewIntoDataStore) {
    constexpr size_t xSize = 6;
    constexpr size_t ySize = 5;
    constexpr size_t zSize = 4;
    double fortranArray[zSize][ySize][xSize];

    gt_fortran_array_descriptor descriptor;
    descriptor.rank = 3;
    descriptor.dims[0] = xSize;
    descriptor.dims[1] = ySize;
    descriptor.dims[2] = zSize;
    descriptor.type = gt_fk_Double;
    descriptor.data = fortranArray;

    gridtools::View< IJKDataStore > fortranArrayView{descriptor};
    IJKDataStore dataStore{IJKStorageInfo{xSize, ySize, zSize}};
    auto dataStoreView = make_host_view(dataStore);

    int i = 0;
    for (size_t z = 0; z < zSize; ++z)
        for (size_t y = 0; y < ySize; ++y)
            for (size_t x = 0; x < xSize; ++x, ++i)
                fortranArray[z][y][x] = i;

    // transform view into dataStore
    transform(dataStore, fortranArrayView);

    i = 0;
    for (size_t z = 0; z < zSize; ++z)
        for (size_t y = 0; y < ySize; ++y)
            for (size_t x = 0; x < xSize; ++x, ++i)
                EXPECT_EQ(dataStoreView(x, y, z), i);
}

TEST(View, TransformDataStoreIntoView) {
    constexpr size_t xSize = 6;
    constexpr size_t ySize = 5;
    constexpr size_t zSize = 4;
    double fortranArray[zSize][ySize][xSize];

    gt_fortran_array_descriptor descriptor;
    descriptor.rank = 3;
    descriptor.dims[0] = xSize;
    descriptor.dims[1] = ySize;
    descriptor.dims[2] = zSize;
    descriptor.type = gt_fk_Double;
    descriptor.data = fortranArray;

    gridtools::View< IJKDataStore > fortranArrayView{descriptor};
    IJKDataStore dataStore{IJKStorageInfo{xSize, ySize, zSize}};
    auto dataStoreView = make_host_view(dataStore);

    int i = 0;
    for (size_t z = 0; z < zSize; ++z)
        for (size_t y = 0; y < ySize; ++y)
            for (size_t x = 0; x < xSize; ++x, ++i)
                dataStoreView(x, y, z) = i;

    // transform dataStore into fortranArrayView
    transform(fortranArrayView, dataStore);

    i = 0;
    for (size_t z = 0; z < zSize; ++z)
        for (size_t y = 0; y < ySize; ++y)
            for (size_t x = 0; x < xSize; ++x, ++i)
                EXPECT_EQ(fortranArray[z][y][x], i);
}
