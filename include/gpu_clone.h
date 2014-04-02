#pragma once
#include <new>
#include "host_device.h"
#include <string>
#include <boost/type_traits/integral_constant.hpp>

namespace gridtools {

#ifdef __CUDACC__
    template <class DerivedType>
    struct mask_object {
        typedef DerivedType type;
        char data[sizeof(DerivedType)];
    };


    template <class T>
    __global__
    void construct(T object) {
        typedef typename T::type derived_type;
        derived_type *p = reinterpret_cast<derived_type*>(&object);
        derived_type* x = new (p->gpu_object_ptr) derived_type(*p);
    }

    template <typename Pointer, typename DerivedType>
    DerivedType* reconstruct(Pointer p, DerivedType const *obj) {
        return new (reinterpret_cast<void*>(p)) DerivedType(*obj);
    }

    /** This class provides methods to copy to and from GPU objects whose data memeners
        are references to other data memebers of the same objects, and other cases in which
        the copy constructor needs to do more than a memcpy. The case in which the object
        contains pointes to heap allocated memory have to be worked out differently.

        A class should publicly derive from this class passing itself as template argument.
        At this point an object of the derived class can call clone_to_gpu() and 
        clone_from_gpu() for doing what should be clear now.
    */
    template <typename DerivedType>
    struct clonable_to_gpu {
        typedef boost::true_type actually_clonable;
        DerivedType* gpu_object_ptr;

        __host__ __device__
        clonable_to_gpu() {
#ifndef __CUDA_ARCH__
            cudaMalloc(&gpu_object_ptr, sizeof(DerivedType));
#endif
        }

        /** Member function to update the object to the gpu calling the copy constructor of the
            derived type.
         */
        void clone_to_gpu() const {
            const mask_object<const DerivedType> *maskT = 
                             reinterpret_cast<const mask_object<const DerivedType>*>
                             ((static_cast<const DerivedType*>(this)));
    
            construct<<<1,1>>>(*maskT);
            cudaDeviceSynchronize();
        }

        /** Member function to update the object from the gpu calling the copy constructor of the
            derived type.
         */
        void clone_from_gpu() {
            mask_object<DerivedType> space;

            cudaMemcpy(&space, gpu_object_ptr, sizeof(DerivedType), cudaMemcpyDeviceToHost);

            DerivedType *x = reconstruct(this, reinterpret_cast<const DerivedType*>(&space));
        }

        ~clonable_to_gpu() {
            cudaFree(gpu_object_ptr);
        }
    };
#else
    template <typename DerivedType>
    struct clonable_to_gpu {
        typedef boost::false_type actually_clonable;
        void clone_to_gpu() const {}
        void clone_from_gpu() const {}
    };
#endif
} // namespace gridtools

