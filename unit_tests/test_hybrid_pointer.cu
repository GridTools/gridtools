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
        p.out();
    }

    __device__
    A(A const& other)
        : p(other.p)
    {
        p.out();
    }
};

__global__
void reverse(A* p, int n) {
    printf("%X ", p->p.cpu_p);
    printf("%X ", p->p.gpu_p);
    printf("%X ", p->p.pointer_to_use);
    printf("%X ", p->p.size);
    printf("\n");
    for (int i = 0; i < p->p.size; ++i)
        p->p[i] = n-i;
}

int main() {
    int n = 10;
    A a(n);

    for (int i = 0; i < n; ++i)
        a.p[i] = i;

    a.p.update_gpu();
    a.clone_to_gpu();

    reverse<<<1,1>>>(a.gpu_object_ptr, n);

    cudaDeviceSynchronize();

    a.p.update_cpu();

    for (int i = 0; i < n; ++i)
        std::cout << a.p[i] << std::endl;

    return 0;
}
