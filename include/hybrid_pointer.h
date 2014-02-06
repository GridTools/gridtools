#pragma once

#include <gt_assert.h>
#include <iostream>

namespace gridtools {
    
    template <typename T>
    struct hybrid_pointer {
        T * cpu_p;
#ifdef __CUDACC__
        T * gpu_p;
#endif
        T * pointer_to_use;
        int size;

        hybrid_pointer(int size) : size(size) {
            allocate_it(size);
            pointer_to_use = cpu_p;
        }

        __host__ __device__
        hybrid_pointer(hybrid_pointer const& other)
            : cpu_p(other.cpu_p)
#ifdef __CUDACC__
            , gpu_p(other.gpu_p)
#endif
#ifdef __CUDA_ARCH__
            , pointer_to_use(gpu_p)
#else
            , pointer_to_use(cpu_p)
#endif
        {} 
         
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
            cpu_p = new T[size];
#ifdef __CUDACC__
#endif
        }

        void free_it() {
#ifdef __CUDACC__
            cudaFree(gpu_p);
#endif
            delete cpu_p;
        }        

        void update_gpu() const {
#ifdef __CUDACC__
            cudaMemcpy(gpu_p, cpu_p, size*sizeof(T), cudaMemcpyHostToDevice);
#endif
        }

        void update_cpu() const {
#ifdef __CUDACC__
            cudaMemcpy(cpu_p, gpu_p, size*sizeof(T), cudaMemcpyDeviceToHost);
#endif
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

            return pointer_to_use[i];
        }

        __host__ __device__
        T const& operator[](int i) const {
            assert(i<size);
            assert(i>=0);

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
