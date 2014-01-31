#include <gpu_clone.h>
#include <stdio.h>
#include <string.h>
#include <hybrid_pointer.h>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

struct B: public gridtools::clonable_to_gpu<B> {
    gridtools::hybrid_pointer<char> pointer_to_use;
    int size;
    int &ref;

    B(int size) : size(size), ref(size), pointer_to_use(size) { }

    __host__ __device__
    B(B const& other)
        : pointer_to_use(other.pointer_to_use)
        , size(other.size)
        , ref(size)
    {}

    ~B() {
        pointer_to_use.free_it();
    }

    void update_gpu() {
        pointer_to_use.update_gpu();
    }

    void update_cpu() {
        pointer_to_use.update_cpu();
    }

};


struct A :public gridtools::clonable_to_gpu<A> {
    int a;
    int &b;
    B p;

    A(int a, int size) : a(a), b(a), p(size) {}

    __host__ __device__
    A(A const& other) 
        : a(other.a)
        , b(a)
        , p(other.p)
    {}
};

#ifdef __CUDACC__
__global__
void test(A* a) {
    printf(">%s<\n", (char*)(a->p.pointer_to_use));
    printf("the reference in A %d\n", a->b);

    a->b++;
    a->p.ref *= 2;
    a->p.pointer_to_use[4] = 'W';
    printf(">%s<\n", &(a->p.pointer_to_use[0]));
}
#endif

void try_char(char *p) {}

int main() {
    char s[30] = "The world will end ... now";

    A a(34, 30);

    // Copy the string to GPU
    memcpy(a.p.pointer_to_use.pointer_to_use, s, 30);
    a.p.update_gpu();

    // Clone a (and b_object) to gpu
    a.clone_to_gpu();

#ifdef __CUDACC__
    test<<<1,1>>>(a.gpu_object_ptr);
#endif
    a.clone_from_gpu();

    printf("%d\n", a.p.size);

    a.p.update_cpu();

    printf("%s\n", (char*)(a.p.pointer_to_use));
    printf("the reference in A %d\n", a.b);

#ifdef __CUDACC__
    cudaDeviceSynchronize();
#endif

    return 0;
}
