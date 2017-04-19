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
#include <errno.h>
#include <iostream>
#include <iomanip>
#include "common/gpu_clone.hpp"
#include "storage/hybrid_pointer.hpp"

using namespace gridtools;

template < typename t_derived >
struct base : public clonable_to_gpu< t_derived > {
    uint_t m_size;

    base(uint_t s) : m_size(s) {}

    __host__ __device__ base(base const &other) // default construct clonable_to_gpu
        : m_size(other.m_size) {}
};

template < typename value_type >
struct derived : public base< derived< value_type > > {
    hybrid_pointer< value_type > data;

    derived(uint_t s) : base< derived< value_type > >(s), data(s) {
        for (uint_t i = 0; i < data.get_size(); ++i)
            data[i] = data.get_size() - i;
        data.update_gpu();
    }

    __host__ __device__ derived(derived const &other) : base< derived< value_type > >(other), data(other.data) {}
};

class clone_derived_args : public ::testing::Test {
  public:
    static uint_t s_buffer_size; // example instance variable

    static void init(uint_t size) {
        s_buffer_size = size;
    } // process argc and argv in this method, retaining such values as your test requires, as with myArgC above
};

uint_t clone_derived_args::s_buffer_size = 0;

int main(int argc, char **argv) {
    uint_t buffer_size = 128;

    if (argc < 2) {
        printf("WARNING: you should pass a buffer size.\n\tUsage: %s [buffer size]\n", argv[0]);
        printf("taking 128 as default.\n");
    } else {
        char *pend = 0;
        buffer_size = strtol(argv[1], &pend, 10);
        if (buffer_size == 0 || pend == 0 || *pend != '\0' || errno == ERANGE) {
            printf("ERROR: invalid buffer size.\n\tUsage: %s [buffer size]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    ::testing::InitGoogleTest(&argc, argv);
    clone_derived_args::init(buffer_size);

    return RUN_ALL_TESTS();
}

TEST_F(clone_derived_args, copy_tests) {
    bool res = true;

    derived< uint_t > a(s_buffer_size);
    for (uint_t i = 0; i < a.data.get_size(); ++i) {
        a.data.update_cpu(); // have to do this before accessing from the CPU (copy back)
        if (a.data[i] != s_buffer_size - i)
            res = false;
    }

    a.clone_to_device();
    a.data.update_gpu();
    a.data.update_cpu();

    for (uint_t i = 0; i < a.data.get_size(); ++i) {
        if (a.data[i] != s_buffer_size - i)
            res = false;
    }

    for (uint_t i = 0; i < a.data.get_size(); ++i) {
        if (a.data[i] != s_buffer_size - i)
            res = false;
    }

    cudaDeviceSynchronize();
    a.clone_from_device();
    a.data.update_cpu();

    for (uint_t i = 0; i < a.data.get_size(); ++i) {
        if (a.data[i] != s_buffer_size - i)
            res = false;
    }

    ASSERT_TRUE(res);
}
