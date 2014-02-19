#pragma once
#include <new>
#include "host_device.h"
#include <string>
namespace gridtools {

#ifdef __CUDACC__
    template <class derived_type>
    struct mask_object {
        typedef derived_type type;
        char data[sizeof(derived_type)];
        
//         mask_object(mask_object const & other) {
//             std::cout << "being copied... " << std::endl;
//             std::cout << std::hex << &other << " " << &(other.data) << std::endl;
//             std::cout << this << " -- " << &(this->data) << std::endl;

//             memcpy(&(this->data), &(other.data), sizeof(derived_type));
//             std::cout << "copied" << std::dec << std::endl;
//             }
    };


    ///QUAAAAAAAAA

    template <class T>
    __global__
    void construct(T object) {
        typedef typename T::type derived_type;
        derived_type *p = reinterpret_cast<derived_type*>(&object);
        derived_type* x = new (p->gpu_object_ptr) derived_type(*p);
    }

    template <typename t_pointer, typename derived_type>
    derived_type* reconstruct(t_pointer p, derived_type const *obj) {
        return new (reinterpret_cast<void*>(p)) derived_type(*obj);
    }

    /** This class provides methods to copy to and from GPU objects whose data memeners
        are references to other data memebers of the same objects, and other cases in which
        the copy constructor needs to do more than a memcpy. The case in which the object
        contains pointes to heap allocated memory have to be worked out differently.

        A class should publicly derive from this class passing itself as template argument.
        At this point an object of the derived class can call clone_to_gpu() and 
        clone_from_gpu() for doing what should be clear now.
    */
    template <typename t_derived_type>
    struct clonable_to_gpu {
        t_derived_type* gpu_object_ptr;

        __host__ __device__
        clonable_to_gpu() {
#ifndef __CUDA_ARCH__
            cudaMalloc(&gpu_object_ptr, sizeof(t_derived_type));
#endif
        }

        /** Member function to update the object to the gpu calling the copy constructor of the
            derived type.
         */
        void clone_to_gpu() const {
#ifndef NDEBUG
            std::cout << std::hex << " ----> "
                      << this << " "
                      << static_cast<const t_derived_type*>(this) <<  std::dec
                      << " * " << sizeof(t_derived_type)
                      << std::endl;
#endif        
            const mask_object<const t_derived_type> *maskT = 
                             reinterpret_cast<const mask_object<const t_derived_type>*>
                             ((static_cast<const t_derived_type*>(this)));
    
#ifndef NDEBUG
            std::cout << "call " 
                      << std::hex << maskT 
                      << std::dec << " " << sizeof(mask_object<const t_derived_type>) 
                      << std::endl;
#endif

            construct<<<1,1>>>(*maskT);
            cudaDeviceSynchronize();
        }

        /** Member function to update the object from the gpu calling the copy constructor of the
            derived type.
         */
        void clone_from_gpu() {
            mask_object<t_derived_type> space;

            cudaMemcpy(&space, gpu_object_ptr, sizeof(t_derived_type), cudaMemcpyDeviceToHost);

            t_derived_type *x = reconstruct(this, reinterpret_cast<const t_derived_type*>(&space));
        }

        ~clonable_to_gpu() {
            cudaFree(gpu_object_ptr);
        }
    };
#else
    template <typename t_derived_type>
    struct clonable_to_gpu {
        void clone_to_gpu() const {}
        void clone_from_gpu() const {}
    };
#endif
} // namespace gridtools

