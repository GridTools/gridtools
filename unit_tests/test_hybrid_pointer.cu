#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <gpu_clone.h>
#include <hybrid_pointer.h>

struct A: gridtools::clonable_to_gpu<A> {
    gridtools::hybrid_pointer<int> p;

    A(int n)
        : p(n)
    {
#ifndef NDEBUG
        p.out();
#endif
    }

    __device__
    A(A const& other)
        : p(other.p)
    {
#ifndef NDEBUG
        p.out();
#endif
    }
};

__global__
void reverse(A* p, int n) {
#ifndef NDEBUG
    printf(" cpu_p %X ", p->p.cpu_p);
    printf(" gpu_p %X ", p->p.gpu_p);
    printf(" to_use %X ", p->p.pointer_to_use);
    printf(" siez %X ", p->p.size);
    printf("\n");
#endif
    for (int i = 0; i < p->p.size; ++i)
        p->p[i] = n-i;
}

bool test_hybrid_pointer() {
    int n = 10;
    A a(n);

    for (int i = 0; i < n; ++i)
        a.p[i] = i;

    a.p.update_gpu();
    a.clone_to_gpu();

    reverse<<<1,1>>>(a.gpu_object_ptr, n);

    cudaDeviceSynchronize();

    a.p.update_cpu();

    bool right = true;
    for (int i = 0; i < n; ++i)
        if (a.p[i] != n-i)
            right = false;

    return right;
}
