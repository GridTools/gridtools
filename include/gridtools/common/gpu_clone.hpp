/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

/**
@file
@brief implements a class to copy to and from GPU objects using the CRTP pattern
*/

namespace gridtools {

    /** \ingroup common
        @{
        \defgroup gpuclone Gpu Cloning
        @{
    */

#if defined(_USE_GPU_) || defined(__CUDACC__)

    /**
       @brief this struct is necessary because otherwise the object would be copied to a temporary storage on the GPU
       (by the construct function), and when deleted the distructor of the base class clonable_to_gpu would be called,
       freeing the memory with cudaFree.
    */
    template <class DerivedType>
    struct mask_object {
        typedef DerivedType type;
        char data[sizeof(DerivedType)];
    };

    /** @brief function called by the device, it allocates the storage on the GPU, given an object on the CPU.
    The template argument T is supposed to be of mask_object type */
    template <class T>
    __global__ void construct(T object) {
        typedef typename T::type derived_type;
        derived_type *p = reinterpret_cast<derived_type *>(&object);
        derived_type *x = new (p->gpu_object_ptr) derived_type(*p);
    }

    /** @brief function called by the host, it copies the object from the GPU back to the CPU
     */
    template <typename Pointer, typename DerivedType>
    DerivedType *reconstruct(Pointer p, DerivedType const *obj) {
        return new (reinterpret_cast<void *>(p)) DerivedType(*obj);
    }

    /** This class provides methods to copy to and from GPU objects whose data members
        are references to other data members of the same objects, and other cases in which
        the copy constructor needs to do more than a memcpy. The case in which the object
        contains pointers to heap allocated memory have to be worked out differently.

        A class should publicly derive from this class passing itself as template argument.
        At this point an object of the derived class can call clone_to_device() and
        clone_from_device() for doing what should be clear now.
    */
    template <typename DerivedType>
    struct clonable_to_gpu {
        typedef boost::true_type actually_clonable;
        typedef DerivedType derived_type;
        derived_type *gpu_object_ptr;

        GT_FUNCTION clonable_to_gpu();

        derived_type *device_pointer() const;
        /** Member function to update the object to the gpu calling the copy constructor of the
            derived type.
         */
        void clone_to_device() const;

        /** Member function to update the object from the gpu calling the copy constructor of the
            derived type.
        */
        void clone_from_device();

        ~clonable_to_gpu();
    };
#else
    template <typename DerivedType>
    struct clonable_to_gpu {
        typedef boost::false_type actually_clonable;
        void clone_to_device() const {}
        void clone_from_device() const {}
    };
#endif

#ifdef __CUDACC__
    template <typename T>
    GT_FUNCTION clonable_to_gpu<T>::clonable_to_gpu() {
#ifndef __CUDA_ARCH__
        cudaMalloc(&gpu_object_ptr, sizeof(clonable_to_gpu<T>::derived_type));
#endif
    }

    template <typename T>
    clonable_to_gpu<T>::derived_type *clonable_to_gpu<T>::device_pointer() const {
        return gpu_object_ptr;
    }
    /** Member function to update the object to the gpu calling the copy constructor of the
        derived type.
    */

    template <typename T>
    void clonable_to_gpu<T>::clone_to_device() const {
        const mask_object<const clonable_to_gpu<T>::derived_type> *maskT =
            reinterpret_cast<const mask_object<const clonable_to_gpu<T>::derived_type> *>(
                (static_cast<const clonable_to_gpu<T>::derived_type *>(this)));

        construct<<<1, 1>>>(*maskT);

        cudaDeviceSynchronize(); // if you want to remove this, then move it in the #ifndef NDEBUG

#ifndef NDEBUG
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA ERROR: %s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
            exit(-1);
        }
#endif
    }

    /** Member function to update the object from the gpu calling the copy constructor of the
        derived type.
    */
    template <typename T>
    void clonable_to_gpu<T>::clone_from_device() {
        mask_object<clonable_to_gpu<T>::derived_type> space;
        cudaMemcpy(&space, gpu_object_ptr, sizeof(clonable_to_gpu<T>::derived_type), cudaMemcpyDeviceToHost);
        clonable_to_gpu<T>::derived_type *x =
            reconstruct(this, reinterpret_cast<const clonable_to_gpu<T>::derived_type *>(&space));
    }

    template <typename T>
    clonable_to_gpu<T>::~clonable_to_gpu() {
        cudaFree(gpu_object_ptr);
        gpu_object_ptr = NULL;
    }
#endif
    /** @} */
    /** @} */
} // namespace gridtools
