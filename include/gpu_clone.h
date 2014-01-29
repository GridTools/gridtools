#pragma once

#include "host_device.h"

namespace gridtools {

#ifdef __CUDACC__
    template <class T>
    struct mask_object {
        char data[sizeof(T)];
    };

    template <class T>
    __global__
    void construct(mask_object<const T> object) {
        T *p = reinterpret_cast<T*>(&object);
        T* x = new (p->gpu_object_ptr) T(*p);
    }

    template <typename T>
    struct clonable_to_gpu {
        T* gpu_object_ptr;

        GT_FUNCTION
        clonable_to_gpu() {
#ifndef __CUDA_ARCH__
            cudaMalloc(&gpu_object_ptr, sizeof(T));
#endif
        }

        void clone_to_gpu() const {
            const mask_object<const T> *maskT = reinterpret_cast<const mask_object<const T>*>((static_cast<const T*>(this)));

            construct<T><<<1,1>>>(*maskT);
            cudaDeviceSynchronize();
        }

        ~clonable_to_gpu() {
            cudaFree(gpu_object_ptr);
        }
    };
#else
    teamplate <typename T>
    struct clonable_to_gpu {
        void clone_to_gpu() const {}
    };
#endif
} // namespace gridtools

