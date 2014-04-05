#pragma once

#include <gt_assert.h>
#include <iostream>
#include <stdio.h>

namespace gridtools {

    namespace workaround_ {
        template <typename T>
        struct new_op;

#define NEW_OP(x) template <>                   \
        struct new_op<x> {                      \
            x* operator()(int size) const {     \
                return new x[size];             \
            }                                   \
        };

        NEW_OP(int)
        NEW_OP(unsigned int)
        NEW_OP(char)
        NEW_OP(float)
        NEW_OP(double)
    }
    
    template <typename T>
    struct hybrid_pointer {
        T * cpu_p;
        T * gpu_p;
        T * pointer_to_use;
        int size;

        hybrid_pointer(int size) : size(size) {
            allocate_it(size);
            pointer_to_use = cpu_p;
#ifndef NDEBUG
            printf(" - %X %X %X %d\n", cpu_p, gpu_p, pointer_to_use, size);
#endif
        }

        __device__
        hybrid_pointer(hybrid_pointer const& other)
            : cpu_p(other.cpu_p)
            , gpu_p(other.gpu_p)
            , pointer_to_use(gpu_p)
            , size(other.size)
        {
#ifndef NDEBUG
            printf("cpy const hp "); 
            printf("%X ", cpu_p);
            printf("%X ", gpu_p);
            printf("%X ", pointer_to_use);
            printf("%d ", size);
            printf("\n");
#endif
        } 

        void allocate_it(int size) {
#ifdef __CUDACC__
            int err = cudaMalloc(&gpu_p, size*sizeof(T));
            if (err != cudaSuccess) {
                std::cout << "Error allocating storage in "
                          << __PRETTY_FUNCTION__
                          << " : size = "
                          << size*sizeof(T)
                          << " bytes "
                          << std::endl;
            }
#endif

#if (CUDA_VERSION > 5050)
            cpu_p = new T[size];
#else
            cpu_p = workaround_::new_op<T>()(size);
#endif
        }

        void free_it() {
#ifdef __CUDACC__
            cudaFree(gpu_p);
#endif
            delete cpu_p;
        }        

        void update_gpu() {
#ifdef __CUDACC__
#ifndef NDEBUG
            printf("update gpu "); out();
#endif
            cudaMemcpy(gpu_p, cpu_p, size*sizeof(T), cudaMemcpyHostToDevice);
#endif
        }

        void update_cpu() {
#ifdef __CUDACC__
#ifndef NDEBUG
            printf("update cpu "); out();
#endif
            cudaMemcpy(cpu_p, gpu_p, size*sizeof(T), cudaMemcpyDeviceToHost);
#endif
        }

        __host__ __device__
        void out() const {
            printf("out hp "); 
            printf("%X ", cpu_p);
            printf("%X ", gpu_p);
            printf("%X ", pointer_to_use);
            printf("%d ", size);
            printf("\n");
        }

        __host__ __device__
        operator T*() {
            return pointer_to_use;
        }

        __host__ __device__
        operator T const*() const {
            return pointer_to_use;
        }

        __host__ __device__
        T& operator[](int i) {
            assert(i<size);
            assert(i>=0);
            // printf(" [%d %e] ", i, pointer_to_use[i]);
            return pointer_to_use[i];
        }

        __host__ __device__
        T const& operator[](int i) const {
            assert(i<size);
            assert(i>=0);
            // printf(" [%d %e] ", i, pointer_to_use[i]);

            return pointer_to_use[i];
        }

        __host__ __device__
        T& operator*() {
            return *pointer_to_use;
        }

        __host__ __device__
        T const& operator*() const {
            return *pointer_to_use;
        }

        __host__ __device__
        T* operator+(int i) {
            return &pointer_to_use[i];
        }

        __host__ __device__
        T* const& operator+(int i) const {
            return &pointer_to_use[i];
        }

    };

} // namespace gridtools
