#pragma once

#include <iostream>
#include <stdio.h>
#include <boost/current_function.hpp>

#include "../common/gt_assert.h"
#include "wrap_pointer.h"

/** @file
    @brief double pointer mapping host and device
    implementation of a double pointer, living on both host and device, together with the algorithms to copy back/to the device. The device is supposed to be a GPU supporting CUDA.
*/
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


/**\todo Note that this struct will greatly simplify when the CUDA arch 3200 and inferior will be obsolete (the "pointer_to_use" will then become useless, and the operators defined in the base class will be usable) */
    template <typename T>
    struct hybrid_pointer : public wrap_pointer<T>{

        explicit hybrid_pointer(T* p) : wrap_pointer<T>(p), gpu_p(NULL), pointer_to_use(p), size(0) {}

        explicit hybrid_pointer(int size) : wrap_pointer<T>(size), size(size) {
            allocate_it(size);
            pointer_to_use = this->cpu_p;
#ifndef NDEBUG
            printf(" - %X %X %X %d\n", this->cpu_p, gpu_p, pointer_to_use, size);
#endif
        }

        __device__
        explicit hybrid_pointer(hybrid_pointer const& other)
            : wrap_pointer<T>(other)
            , gpu_p(other.gpu_p)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 3200)
            , pointer_to_use(gpu_p)
#else
            , pointer_to_use(this->cpu_p)
#endif
            , size(other.size)
        {
#ifndef NDEBUG
            printf("cpy const hp ");
            printf("%X ", this->cpu_p);
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
                          << BOOST_CURRENT_FUNCTION
                          << " : size = "
                          << size*sizeof(T)
                          << " bytes "
                          << std::endl;
            }
#endif
        }

        void free_it() {
#ifdef __CUDACC__
	  cudaFree(gpu_p);
#endif
	  wrap_pointer<T>::free_it();
        }

        void update_gpu() {
#ifdef __CUDACC__
#ifndef NDEBUG
            printf("update gpu "); out();
#endif
            cudaMemcpy(gpu_p, this->cpu_p, size*sizeof(T), cudaMemcpyHostToDevice);
#endif
        }

        void update_cpu() {
#ifdef __CUDACC__
#ifndef NDEBUG
            printf("update cpu "); out();
#endif
            cudaMemcpy(this->cpu_p, gpu_p, size*sizeof(T), cudaMemcpyDeviceToHost);
#endif
        }

        __host__ __device__
        void out() const {
            printf("out hp ");
            printf("%X ", this->cpu_p);
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
            /* assert(i<size); */
            /* assert(i>=0); */
            // printf(" [%d %e] ", i, pointer_to_use[i]);
            return pointer_to_use[i];
        }

        __host__ __device__
        T const& operator[](int i) const {
            /* assert(i<size); */
            /* assert(i>=0); */
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

        GT_FUNCTION
        T* get_gpu_p(){return gpu_p;};

        GT_FUNCTION
        T* get_pointer_to_use(){return pointer_to_use;}

        GT_FUNCTION
        int get_size(){return size;}

    private:
        T * gpu_p;
        T * pointer_to_use;
        int size;


    };

} // namespace gridtools
