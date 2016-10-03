/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "gtest/gtest.h"
#include <storage/data_field.hpp>
#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace gridtools::enumtype;

TEST(storage, test_data_field) {
#ifdef STRUCTURED_GRIDS
    typedef base_storage< wrap_pointer< int >, backend< Host, structured, Naive >::storage_info< 0, layout_map< 0, 1 > > >
        storage_t;
    backend< Host, structured, Naive >::storage_info< 0, layout_map< 0, 1 > > meta_(1, 1);

    field< storage_t, 3, 2, 4 >::type datafield(&meta_, 0, "data");

    datafield.get_value< 1, 0 >(0, 0) = 1;
    datafield.get_value< 2, 0 >(0, 0) = 2;

    datafield.get_value< 0, 1 >(0, 0) = 10;
    datafield.get_value< 1, 1 >(0, 0) = 11;

    datafield.get_value< 0, 2 >(0, 0) = 100;
    datafield.get_value< 1, 2 >(0, 0) = 101;
    datafield.get_value< 2, 2 >(0, 0) = 102;
    datafield.get_value< 3, 2 >(0, 0) = 103;

    /*swaps the first and last snapshots of the first dimension*/
    swap< 0, 0 >::with< 2, 0 >::apply(datafield);
    assert((datafield.get_value< 0, 0 >(0, 0) == 2 && datafield.get_value< 2, 0 >(0, 0) == 0));

    std::cout << "STORAGE VALUES BEFORE: " << datafield.get_value< 0, 2 >(0, 0) << " "
              << datafield.get_value< 1, 2 >(0, 0) << " " << datafield.get_value< 2, 2 >(0, 0) << " "
              << datafield.get_value< 3, 2 >(0, 0) << std::endl;
#endif
    ASSERT_TRUE(true);
}
