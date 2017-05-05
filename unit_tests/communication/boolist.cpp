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
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#include <common/boollist.hpp>
#include <common/layout_map.hpp>
#include <iostream>
#include <stdlib.h>
#include "gtest/gtest.h"

TEST(Communication, boollist) {

    bool pass = true;

    for (int i = 0; i < 100000; ++i) {
        bool v0 = rand() % 2, v1 = rand() % 2, v2 = rand() % 2;

        gridtools::boollist< 3 > bl1(v0, v1, v2);

        if (bl1.value(0) != v0)
            pass = false;

        if (bl1.value(1) != v1)
            pass = false;

        if (bl1.value(2) != v2)
            pass = false;

        gridtools::boollist< 3 > bl2 = bl1.permute< gridtools::layout_map< 1, 2, 0 > >();

        if (bl2.value(0) != v2)
            pass = false;

        if (bl2.value(1) != v0)
            pass = false;

        if (bl2.value(2) != v1)
            pass = false;

        gridtools::boollist< 3 > bl3 = bl1.permute< gridtools::layout_map< 2, 1, 0 > >();

        if (bl3.value(0) != v2)
            pass = false;

        if (bl3.value(1) != v1)
            pass = false;

        if (bl3.value(2) != v0)
            pass = false;

        gridtools::boollist< 3 > bl4 = bl1.permute< gridtools::layout_map< 0, 1, 2 > >();

        if (bl4.value(0) != v0)
            pass = false;

        if (bl4.value(1) != v1)
            pass = false;

        if (bl4.value(2) != v2)
            pass = false;
    }

    EXPECT_TRUE(pass);
}
