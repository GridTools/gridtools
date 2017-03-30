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

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "common/gpu_clone.hpp"
#include "storage/hybrid_pointer.hpp"

using gridtools::uint_t;
using gridtools::int_t;

struct A : gridtools::clonable_to_gpu< A > {
    gridtools::hybrid_pointer< int > p;

    A(uint_t n) : p(n) {
#ifndef NDEBUG
        p.out();
#endif
    }

    __device__ A(A const &other) : p(other.p) {
#ifndef NDEBUG
        p.out();
#endif
    }
};

__global__ void reverse(A *p, uint_t n) {
#ifndef NDEBUG
    if (p->p.on_host())
        printf(" cpu_p %X ", p->p.get_cpu_p());
    if (p->p.on_device())
        printf(" gpu_p %X ", p->p.get_gpu_p());
    printf(" to_use %X ", p->p.get_pointer_to_use());
    printf(" siez %X ", p->p.get_size());
    printf("\n");
#endif
    for (uint_t i = 0; i < p->p.get_size(); ++i)
        p->p[i] = n - i;
}

bool test_hybrid_pointer() {
    uint_t n = 10;
    A a(n);

    for (uint_t i = 0; i < n; ++i)
        a.p[i] = i;

    a.p.update_gpu();
    a.clone_to_device();

    // clang-format off
    reverse<<<1,1>>>(a.gpu_object_ptr, n);
    // clang-format on

    cudaDeviceSynchronize();

    a.p.update_cpu();

    bool right = true;
    for (uint_t i = 0; i < n; ++i)
        if (a.p[i] != n - i)
            right = false;

    return right;
}

TEST(test_hybrid_pointer, hybrid_pointer_on_gpu) { EXPECT_EQ(test_hybrid_pointer(), true); }
