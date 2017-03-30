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
#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>

namespace test_storage_info_gpu_using {

    using namespace gridtools;
    typedef layout_map< 0, 1, 2 > layout_t;
    typedef meta_storage< meta_storage_base< static_int< 0 >, layout_t, false > > meta_t;
    typedef storage< base_storage< hybrid_pointer< double >, meta_t > > storage_t;

    template < typename T >
    __global__ void set(T *st_) {

        for (int i = 0; i < 11; ++i)
            for (int j = 0; j < 12; ++j)
                for (int k = 0; k < 13; ++k)
                    // st_->fields()[0].out();
                    // printf("(*st_)(i,j,k) = %d", (*st_)(i,j,k));
                    (*st_)(i, j, k) = (double)i + j + k;
    }

    TEST(storage_info, test_pointer) {
        meta_t meta_(11, 12, 13);
        storage_t st_(meta_, 5.);
        st_.h2d_update();
        st_.clone_to_device();

        // clang-format off
    set<<<1,1>>>(st_.get_pointer_to_use());
        // clang-format on

        st_.d2h_update();
        // st_.print();

        bool ret = true;
        for (int i = 0; i < 11; ++i)
            for (int j = 0; j < 12; ++j)
                for (int k = 0; k < 13; ++k)
                    if (st_(i, j, k) != (double)i + j + k) {
                        ret = false;
                    }

        ASSERT_TRUE(ret);
    }

} // namespace test_storage_info_gpu_pointer
