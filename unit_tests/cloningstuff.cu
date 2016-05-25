#include "common/gpu_clone.hpp"
#include <stdio.h>
#include <string.h>
#include "storage/hybrid_pointer.hpp"
#include <algorithm>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

using gridtools::uint_t;
using gridtools::int_t;

namespace cloningstuff_test {
    struct B: public gridtools::clonable_to_gpu<B> {
        gridtools::hybrid_pointer<char> pointer_to_use;
        uint_t size;
        uint_t &ref;

        B(uint_t size) : size(size), ref(size), pointer_to_use(size) { }

        __host__ __device__
        B(B const& other)
            : pointer_to_use(other.pointer_to_use)
            , size(other.size)
            , ref(size)
        {}

        ~B() {
            //pointer_to_use.free_it();
        }

        void update_gpu() {
            pointer_to_use.update_gpu();
        }

        void update_cpu() {
            pointer_to_use.update_cpu();
        }

    };


    struct A :public gridtools::clonable_to_gpu<A> {
        uint_t a;
        uint_t &b;
        B p;

        A(uint_t a, uint_t size) : a(a), b(a), p(size) {}

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
        // printf(">%s<\n", (char*)(a->p.pointer_to_use));
        // printf("the reference in A %d\n", a->b);

        a->b++;
        a->p.ref *= 2;
        a->p.pointer_to_use[4] = 'W';
        // printf(">%s<\n", &(a->p.pointer_to_use[0]));
    }
#endif

    void try_char(char *p) {}

    bool test_cloningstuff() {
        char s[30] = "The world will end ... now";
        A a(34, 30);

        // Copy the string to GPU
        memcpy(a.p.pointer_to_use.get_pointer_to_use(), s, 30);
        a.p.update_gpu();

        // Clone a (and b_object) to gpu
        a.clone_to_device();

#ifdef __CUDACC__
        // clang-format off
        test<<<1,1>>>(a.gpu_object_ptr);
        // clang-format on
        cudaDeviceSynchronize();
#endif
        a.clone_from_device();

        bool result = true;

        result = (a.p.size == 60);

        a.p.update_cpu();

        if (strcmp(static_cast<char*>(a.p.pointer_to_use), "The World will end ... now") != 0) {
            std::cout << "here" << std::endl;
            result = false;
        }

        if (a.b != 35)
            result = false;

        return result;
    }
} // namespace cloningstuff_test
