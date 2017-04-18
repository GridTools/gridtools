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
#include <iostream>
#include <string>

#include <stencil-composition/stencil-composition.hpp>

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

TEST(storage, naming) {
    typedef gridtools::layout_map< 0, 1, 2 > layout_t;
    typedef backend< Host, GRIDBACKEND, Naive >::storage_info< __COUNTER__, layout_t > meta_data_t;
    typedef backend< Host, GRIDBACKEND, Naive >::storage_type< float_type, meta_data_t >::type storage_t;

    meta_data_t meta_data_(1, 1, 1);

    typedef storage_t storage_type;
    // create storage with name in
    storage_type in(meta_data_, "bla");
    EXPECT_EQ(std::string(in.get_name()), "bla");
    // set name to modified using const char *
    in.set_name("modified_ccp");
    EXPECT_EQ(std::string(in.get_name()), "modified_ccp");
    // set name using string that gets deleted afterwards
    {
        std::string t("strtmp");
        in.set_name(t.c_str());
        // t gets destroyed here
    }
    {
        std::string t("tmpstr"); // value of in.m_name maybe gets overridden here by accident
        // validate it is still the old string
        EXPECT_EQ(std::string(in.get_name()), "strtmp");
    }
}
