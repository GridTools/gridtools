/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
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
