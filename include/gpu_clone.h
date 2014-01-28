#pragma once
#include "host_device.h" 

namespace gridtools {

    template <class T>
    struct mask_object {
        char data[sizeof(T)];
    };

    template <class T>
    __global__
    void construct(mask_object<T> object) {
        T *p = reinterpret_cast<T*>(&object);
        T* x = new (p->gpu_object_ptr) T(*p);
    }

    template <typename T>
    struct gpu_clone {
        T* gpu_object_ptr;

        GT_FUNCTION
        gpu_clone() {
#ifndef __CUDA_ARCH__
            cudaMalloc(&gpu_object_ptr, sizeof(T));
#endif
        }

        void clone() {
            mask_object<T> *maskT = reinterpret_cast<mask_object<T>*>((static_cast<T*>(this)));

            construct<T><<<1,1>>>(*maskT);
            cudaDeviceSynchronize();
        }

        ~gpu_clone() {
            cudaFree(gpu_object_ptr);
        }
    };

} // namespace gridtools

