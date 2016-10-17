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

    A(uint_t n) : p(n, false) {
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
