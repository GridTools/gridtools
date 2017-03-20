/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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
#include <common/cuda_type_traits.hpp>

TEST(texture_type_traits, int_is_texture_type) { ASSERT_TRUE(gridtools::is_texture_type< int >::value); }

TEST(texture_type_traits, bool_is_NOT_texture_type) { ASSERT_FALSE(gridtools::is_texture_type< bool >::value); }

TEST(texture_type_traits, real_typedef_is_texture_type) {
    typedef double Real;
    ASSERT_TRUE(gridtools::is_texture_type< Real >::value);
}

TEST(texture_type_traits, gridtools_uint_is_texture_type) {
    ASSERT_TRUE(gridtools::is_texture_type< gridtools::uint_t >::value);
}

TEST(texture_type_traits, int_ref_is_texture_type) { ASSERT_TRUE(gridtools::is_texture_type< int & >::value); }

TEST(texture_type_traits, cv_int_is_texture_type) {
    ASSERT_TRUE(gridtools::is_texture_type< const volatile int >::value);
}

TEST(texture_type_traits, restrict_int_ref_is_texture_type) {
    // We need this typedef for clang to work as CUDA host compiler
    typedef gridtools::is_texture_type< int &__restrict__ >::type type;
    ASSERT_TRUE((type::value));
}

TEST(texture_type_traits, restrict_int_ptr_is_texture_type) {
    ASSERT_TRUE(gridtools::is_texture_type< int *__restrict__ >::value);
}

#ifdef CXX11_ENABLED
TEST(texture_type_traits, is_texture_type_t) {
    using result = gridtools::is_texture_type_t< int >;
    ASSERT_TRUE(result::value);
}
#endif
