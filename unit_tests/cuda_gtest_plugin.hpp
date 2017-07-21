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
#pragma once

#include "gtest/gtest.h"
#ifdef __CUDACC__

#ifndef CUDA_LAST_ERROR
#define CUDA_LAST_ERROR(msg)                                                    \
    {                                                                           \
        cudaDeviceSynchronize();                                                \
        cudaError_t error = cudaGetLastError();                                 \
        if (error != cudaSuccess) {                                             \
            fprintf(stderr, "ERROR: %s: %s\n", msg, cudaGetErrorString(error)); \
            exit(-1);                                                           \
        }                                                                       \
    }
#endif

struct TestTransporter {
    float2 tfloat[10];
    int2 tint[10];

    int evaluateInt;
    int evaluateFloat;

    __host__ __device__ TestTransporter() : evaluateInt(0), evaluateFloat(0){};
};

template < typename T >
static __host__ __device__ void setTestTransporterValue(TestTransporter *transporter, T expected, T actual);

template <>
__host__ __device__ void setTestTransporterValue(TestTransporter *transporter, float expected, float actual) {
    transporter->tfloat[transporter->evaluateFloat].x = expected;
    transporter->tfloat[transporter->evaluateFloat].y = actual;
    transporter->evaluateFloat++;
}

template <>
__host__ __device__ void setTestTransporterValue(TestTransporter *transporter, int expected, int actual) {
    transporter->tint[transporter->evaluateInt].x = expected;
    transporter->tint[transporter->evaluateInt].y = actual;
    transporter->evaluateInt++;
}

template <>
__host__ __device__ void setTestTransporterValue(TestTransporter *transporter, bool expected, bool actual) {
    transporter->tint[transporter->evaluateInt].x = (int)expected;
    transporter->tint[transporter->evaluateInt].y = (int)actual;
    transporter->evaluateInt++;
}

#define CUDA_TEST_CLASS_NAME_(test_case_name, test_name) kernel_test_case_name##_##test_name##_Test

#ifdef __CUDA_ARCH__
#undef TEST
#define CUDA_DEAD_FUNCTION_NAME_(test_case_name, test_name) \
    MAKE_UNIQUE(dead_function_test_case_name##_##test_name##_Test)
#define TEST(test_case_name, test_name)                       \
    void CUDA_DEAD_FUNCTION_NAME_(test_case_name, test_name)( \
        TestTransporter * testTransporter) // GTEST_TEST(test_case_name, test_name)
#define TESTTRANSPORTERDEFINITIONWITHCOMMA , TestTransporter *testTransporter
#define TESTTRANSPORTERDEFANDINSTANCE
#define TESTTRANSPORTERDEFINITION TestTransporter *testTransporter
#define TESTCALLHOST
#define TESTCALLDEVICE test(testTransporter)
#else
#define CUDA_DEAD_FUNCTION_NAME_(test_case_name, test_name)
#define TESTTRANSPORTERDEFANDINSTANCE TestTransporter *testTransporter = new TestTransporter;
#define TESTTRANSPORTERDEFINITIONWITHCOMMA
#define TESTTRANSPORTERDEFINITION
#define TESTCALLHOST test()
#define TESTCALLDEVICE
#endif

#define TESTKERNELCALL(test_case_name, test_name)             \
    CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) test; \
    CUDA_TEST_CLASS_NAME_(test_case_name, test_name)<<< 1, 1 >>>(test, dTestTransporter)

#define CUDA_ASSERT_EQ(expected, actual) setTestTransporterValue(testTransporter, expected, actual);

#ifdef __CUDA_ARCH__
#undef ASSERT_EQ
#define ASSERT_EQ(val1, val2) CUDA_ASSERT_EQ(val1, val2)
#endif

#ifdef __CUDA_ARCH__
#undef ASSERT_FLOAT_EQ
#define ASSERT_FLOAT_EQ(val1, val2) CUDA_ASSERT_EQ(val1, val2)
#endif

#define CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) test_function_test_case_name##_##test_name##_Test
#define TEST_NAME_CUDA(test_name) test_name##_CUDA

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define MAKE_UNIQUE(x) CONCATENATE(x, __COUNTER__)

#define CUDA_TEST(test_case_name, test_name)                                                            \
    struct CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) {                                        \
        __host__ __device__ void operator()(TestTransporter * testTransporter);                         \
    };                                                                                                  \
    __global__ void CUDA_TEST_CLASS_NAME_(test_case_name, test_name)(                                   \
        CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) test, TestTransporter * testTransporter);   \
    GTEST_TEST(test_case_name, test_name) {                                                             \
        CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) test;                                       \
        TestTransporter *testTransporter = new TestTransporter;                                         \
        test(testTransporter);                                                                          \
    };                                                                                                  \
    TEST(test_case_name, test_name##_CUDA) {                                                            \
        TestTransporter *dTestTransporter;                                                              \
        cudaMalloc((void **)(&dTestTransporter), sizeof(TestTransporter));                              \
        CUDA_LAST_ERROR("malloc");                                                                      \
        TESTTRANSPORTERDEFANDINSTANCE                                                                   \
        cudaMemcpy(dTestTransporter, testTransporter, sizeof(TestTransporter), cudaMemcpyHostToDevice); \
        CUDA_LAST_ERROR("memcopyhosttodevice");                                                         \
        CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) test;                                       \
        CUDA_TEST_CLASS_NAME_(test_case_name, test_name)<<< 1, 1 >>>(test, dTestTransporter);       \
        CUDA_LAST_ERROR("kernel call");                                                                 \
        cudaMemcpy(testTransporter, dTestTransporter, sizeof(TestTransporter), cudaMemcpyDeviceToHost); \
        CUDA_LAST_ERROR("memcopydevicetohost");                                                         \
        for (int i = 0; i < testTransporter->evaluateFloat; i++)                                        \
            ASSERT_FLOAT_EQ(testTransporter->tfloat[i].x, testTransporter->tfloat[i].y);                \
        for (int i = 0; i < testTransporter->evaluateInt; i++)                                          \
            GTEST_ASSERT_EQ(testTransporter->tint[i].x, testTransporter->tint[i].y);                    \
    };                                                                                                  \
    __global__ void CUDA_TEST_CLASS_NAME_(test_case_name, test_name)(                                   \
        CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) test, TestTransporter * testTransporter) {  \
        test(testTransporter);                                                                          \
    }                                                                                                   \
    __host__ __device__ void CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name)::operator()(           \
        TestTransporter * testTransporter)
#else
#define CUDA_TEST(test_case_name, test_name) TEST(test_case_name, test_name)
#endif
