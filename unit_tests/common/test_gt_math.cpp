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
#include <gridtools/common/defs.hpp>
#include <gridtools/common/gt_math.hpp>
#include <gridtools/tools/verifier.hpp>
using namespace gridtools;

template <typename Value>
struct test_pow {
    static bool GT_FUNCTION apply(Value val, Value result) {
        return expect_with_threshold(math::pow(val, val), result);
    }
};

template <typename Value>
struct test_log {
    static bool GT_FUNCTION apply(Value val, Value result) { return expect_with_threshold(math::log(val), result); }
};

template <typename Value>
struct test_exp {
    static bool GT_FUNCTION apply(Value val, Value result) { return expect_with_threshold(math::exp(val), result); }
};

struct test_fabs {
    static bool GT_FUNCTION apply() {
        GT_STATIC_ASSERT((std::is_same<decltype(math::fabs(4.0f)), float>::value), "Should return float.");
        GT_STATIC_ASSERT((std::is_same<decltype(math::fabs(4.0)), double>::value), "Should return double.");
#ifndef __CUDA_ARCH__
        GT_STATIC_ASSERT(
            (std::is_same<decltype(math::fabs((long double)4)), long double>::value), "Should return long double.");
#endif
        GT_STATIC_ASSERT((std::is_same<decltype(math::fabs((int)4)), double>::value), "Should return double.");

        if (!expect_with_threshold(math::fabs(5.6), 5.6, 1e-14))
            return false;
        else if (!expect_with_threshold(math::fabs(-5.6), 5.6, 1e-14))
            return false;
        else if (!expect_with_threshold(math::fabs(-5.6f), 5.6f, 1e-14))
            return false;
        else if (!expect_with_threshold(math::fabs(-5), (double)5, 1e-14))
            return false;
#ifndef __CUDA_ARCH__
        else if (!expect_with_threshold(math::fabs((long double)-5), (long double)5., 1e-14))
            return false;
#endif
        else
            return true;
    }
};

struct test_abs {
    static GT_FUNCTION bool apply() {
        // float overloads
        GT_STATIC_ASSERT((std::is_same<decltype(math::abs(4.0f)), float>::value), "Should return float.");
        GT_STATIC_ASSERT((std::is_same<decltype(math::abs(4.0)), double>::value), "Should return double.");
#ifndef __CUDA_ARCH__
        GT_STATIC_ASSERT(
            (std::is_same<decltype(math::abs((long double)4)), long double>::value), "Should return long double.");
#endif

        // int overloads
        GT_STATIC_ASSERT((std::is_same<decltype(math::abs((int)4)), int>::value), "Should return int.");
        GT_STATIC_ASSERT((std::is_same<decltype(math::abs((long)4)), long>::value), "Should return long.");
        GT_STATIC_ASSERT(
            (std::is_same<decltype(math::abs((long long)4)), long long>::value), "Should return long long.");

        if (math::abs(5.6) != 5.6)
            return false;
        else if (math::abs(-5.6) != 5.6)
            return false;
        else if (math::abs(-5.6f) != 5.6f)
            return false;
        else if (math::abs(-5) != 5)
            return false;
        else
            return true;
    }
};

TEST(math, test_min) {
    EXPECT_TRUE(math::min(5, 2, 7) == 2);
    EXPECT_TRUE(math::min(5, -1) == -1);

    ASSERT_EQ(math::min(5.3, 22.0, 7.7), 5.3);
}

#ifdef __INTEL_COMPILER
// test disabled due to Intel compiler bug (see gt_math.hpp)
TEST(math, DISABLED_test_min_ref) {
#else
TEST(math, test_min_ref) {
#endif
    // checking returned by const &
    double a = 3.5;
    double b = 2.3;
    double const &min = math::min(a, b);
    ASSERT_EQ(min, 2.3);
    b = 8;
    ASSERT_EQ(min, 8);
}

TEST(math, test_max) {
    EXPECT_TRUE(math::max(5, 2, 7) == 7);
    EXPECT_TRUE(math::max(5, -1) == 5);

    ASSERT_EQ(math::max(5.3, 22.0, 7.7), 22.0);
}

#ifdef __INTEL_COMPILER
// test disabled due to Intel compiler bug (see gt_math.hpp)
TEST(math, DISABLED_test_max_ref) {
#else
TEST(math, test_max_ref) {
#endif
    // checking returned by const &
    double a = 3.5;
    double b = 2.3;
    double const &max = math::max(a, b);

    ASSERT_EQ(max, 3.5);
    a = 8;
    ASSERT_EQ(max, 8);
}

TEST(math, test_fabs) { EXPECT_TRUE(test_fabs::apply()); }

TEST(math, test_abs) { EXPECT_TRUE(test_abs::apply()); }

TEST(math, test_log) {
    EXPECT_TRUE(test_log<double>::apply(2.3, std::log(2.3)));
    EXPECT_TRUE(test_log<float>::apply(2.3f, std::log(2.3f)));
}

TEST(math, test_exp) {
    EXPECT_TRUE(test_exp<double>::apply(2.3, std::exp(2.3)));
    EXPECT_TRUE(test_exp<float>::apply(2.3f, std::exp(2.3f)));
}

TEST(math, test_pow) {
    EXPECT_TRUE(test_pow<double>::apply(2.3, std::pow(2.3, 2.3)));
    EXPECT_TRUE(test_pow<float>::apply(2.3f, std::pow(2.3f, 2.3f)));
}

TEST(math, test_fmod) {
    EXPECT_FLOAT_EQ(math::fmod(3.7f, 1.2f), std::fmod(3.7f, 1.2f));
    EXPECT_DOUBLE_EQ(math::fmod(3.7, 1.2), std::fmod(3.7, 1.2));
    EXPECT_DOUBLE_EQ(math::fmod(3.7l, 1.2l), std::fmod(3.7l, 1.2l));
}

TEST(math, test_trunc) {
    EXPECT_FLOAT_EQ(math::trunc(3.7f), std::trunc(3.7f));
    EXPECT_DOUBLE_EQ(math::trunc(3.7), std::trunc(3.7));
    EXPECT_DOUBLE_EQ(math::trunc(3.7l), std::trunc(3.7l));
}
