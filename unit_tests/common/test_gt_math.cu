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
#include "test_gt_math.cpp"

#include "../cuda_test_helper.hpp"

TEST(math_cuda, test_fabs) {
    EXPECT_TRUE(test_fabs::apply());
    EXPECT_TRUE(cuda_test<test_fabs>());
}

TEST(math_cuda, test_abs) {
    EXPECT_TRUE(test_fabs::apply());
    EXPECT_TRUE(cuda_test<test_fabs>());
}

TEST(math_cuda, test_log) {
    EXPECT_TRUE(test_log<double>::apply(2.3, std::log(2.3)));
    EXPECT_TRUE(test_log<float>::apply(2.3f, std::log(2.3f)));

    EXPECT_TRUE(cuda_test<test_log<double>>(2.3, std::log(2.3)));
    EXPECT_TRUE(cuda_test<test_log<float>>(2.3f, std::log(2.3f)));
}

TEST(math_cuda, test_exp) {

    EXPECT_TRUE(test_exp<double>::apply(2.3, std::exp(2.3)));
    EXPECT_TRUE(test_exp<float>::apply(2.3f, std::exp(2.3f)));

    EXPECT_TRUE(cuda_test<test_exp<double>>(2.3, std::exp(2.3)));
    EXPECT_TRUE(cuda_test<test_exp<float>>(2.3f, std::exp(2.3f)));
}

TEST(math_cuda, test_pow) {

    EXPECT_TRUE(test_pow<double>::apply(2.3, std::pow(2.3, 2.3)));
    EXPECT_TRUE(test_pow<float>::apply(2.3f, std::pow(2.3f, 2.3f)));

    EXPECT_TRUE(cuda_test<test_pow<double>>(2.3, std::pow(2.3, 2.3)));
    EXPECT_TRUE(cuda_test<test_pow<float>>(2.3f, std::pow(2.3f, 2.3f)));
}
