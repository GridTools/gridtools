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
#include <common/defs.hpp>
#include <storage/storage-facility.hpp>
#include <stencil-composition/tile.hpp>
#include <stencil-composition/array_tuple.hpp>

using namespace gridtools;

TEST(tmp_storage_info, test_initialize) {

    typedef gridtools::meta_storage_base< static_int< 0 >, layout_map< 0, 1, 2 >, true > meta_t;

    uint_t d3 = 10;
    uint_t blocks_i = 4;
    uint_t blocks_j = 5;
    typedef meta_storage_tmp< meta_t, tile< 5, 1, 1 >, tile< 3, 3, 3 > > meta_storage_t;
    meta_storage_t instance_(d3, blocks_i, blocks_j);

    array< int_t, 2 > strides_;
    strides_[0] = d3 * 9 * 5;
    strides_[1] = d3;

    ASSERT_TRUE((strides_[0] == instance_.template strides< 0 >()) && "error");
    ASSERT_TRUE((strides_[1] == instance_.template strides< 1 >()) && "error");

    bool success = true;
    for (int_t i = 0; i < blocks_i; ++i)
        for (int_t j = 0; j < blocks_j; ++j) {
            int_t index_ = 0;
            instance_.initialize< 0 >(0, i, &index_, strides_);
            instance_.initialize< 1 >(0, j, &index_, strides_);
            if (index_ !=
                (0 - i * 5
#ifdef __CUDACC__ // TODO keep the cuda version
                    +
                    i * 7
#endif
                    ) * strides_[0] +
                    (0 - j * 3
#ifdef __CUDACC__ // TODO keep the cuda version
                        +
                        j * 9
#endif
                        ) *
                        strides_[1]) {
                std::cout << "Host: [" << i << j << "]" << index_
                          << " != " << -(i * 5) * strides_[0] - (j * 3) * strides_[1] << "\n";
                std::cout << "Cuda: [" << i << j << "]" << index_
                          << " != " << (-i * 5 + i * 7) * strides_[0] + (-j * 3 + j * 9) * strides_[1] << "\n";
                success = false;
            }
        }
    ASSERT_TRUE(success);
}
