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
#include <cstdlib>
#include "common/defs.hpp"
#include "common/atomic_functions.hpp"

template < typename T >
struct Verifier {
    static void TestEQ(T val, T exp) {
        T err = std::fabs(val - exp) / std::fabs(val);
        ASSERT_TRUE(err < 1e-12);
    }
};

template <>
struct Verifier< float > {
    static void TestEQ(float val, float exp) {
        double err = std::fabs(val - exp) / std::fabs(val);
        ASSERT_TRUE(err < 1e-6);
    }
};

template <>
struct Verifier< int > {
    static void TestEQ(int val, int exp) { ASSERT_EQ(val, exp); }
};

template < typename T >
void TestAtomicAdd() {
    int size = 360;
    T field[size];
    T sum = 0;
    T sumRef = 0;
    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast< T >(std::rand() % 100 + (std::rand() % 100) * 0.005);
        sumRef += field[cnt];
    }

#pragma omp for nowait
    for (int cnt = 0; cnt < size; ++cnt) {
        gridtools::atomic_add(sum, field[cnt]);
    }
    ASSERT_REAL_EQ(sumRef, sum);
}

template < typename T >
void TestAtomicSub() {
    int size = 360;
    T field[size];
    T sum = 0;
    T sumRef = 0;
    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast< T >(std::rand() % 100 + (std::rand() % 100) * 0.005);
        sumRef -= field[cnt];
    }

#pragma omp for nowait
    for (int cnt = 0; cnt < size; ++cnt) {
        gridtools::atomic_sub(sum, field[cnt]);
    }
    ASSERT_REAL_EQ(sumRef, sum);
}

template < typename T >
void TestAtomicMin() {
    int size = 360;
    T field[size];
    T min = 99999;
    T minRef = 99999;
    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast< T >(std::rand() % 100 + (std::rand() % 100) * 0.005);
        minRef = std::min(minRef, field[cnt]);
    }

#pragma omp for nowait
    for (int cnt = 0; cnt < size; ++cnt) {
        gridtools::atomic_min(min, field[cnt]);
    }
    Verifier< T >::TestEQ(minRef, min);
}

template < typename T >
void TestAtomicMax() {
    int size = 360;
    T field[size];
    T max = 0;
    T maxRef = 0;
    for (int cnt = 0; cnt < size; ++cnt) {
        field[cnt] = static_cast< T >(std::rand() % 100 + (std::rand() % 100) * 0.005);
        maxRef = std::max(maxRef, field[cnt]);
    }

#pragma omp for nowait
    for (int cnt = 0; cnt < size; ++cnt) {
        gridtools::atomic_max(max, field[cnt]);
    }
    ASSERT_REAL_EQ(maxRef, max);
}

TEST(AtomicFunctionsUnittest, add) {
    TestAtomicAdd< int >();
    TestAtomicAdd< double >();
    TestAtomicAdd< float >();
}

TEST(AtomicFunctionsUnittest, sub) {
    TestAtomicSub< int >();
    TestAtomicSub< double >();
    TestAtomicSub< float >();
}

TEST(AtomicFunctionsUnittest, min) {
    TestAtomicMin< int >();
    TestAtomicMin< double >();
    TestAtomicMin< float >();
}

TEST(AtomicFunctionsUnittest, max) {
    TestAtomicMax< int >();
    TestAtomicMax< double >();
    TestAtomicMax< float >();
}
