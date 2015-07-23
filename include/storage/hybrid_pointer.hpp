#pragma once

#include <iostream>
#include <stdio.h>
#include <boost/current_function.hpp>

#include "../common/gt_assert.hpp"
#include "wrap_pointer.hpp"

/** @file
    @brief double pointer mapping host and device
    implementation of a double pointer, living on both host and device, together with the algorithms to copy back/to the device. The device is supposed to be a GPU supporting CUDA.
*/
namespace gridtools {

/**\todo Note that this struct will greatly simplify when the CUDA arch 3200 and inferior will be obsolete (the "pointer_to_use" will then become useless, and the operators defined in the base class will be usable) */
    template <typename T>
    struct hybrid_pointer : public wrap_pointer<T>{

        typedef wrap_pointer<T> super;
        typedef typename super::pointee_t pointee_t;

        GT_FUNCTION
        explicit  hybrid_pointer() : wrap_pointer<T>((T*)NULL), m_gpu_p(NULL), m_pointer_to_use(NULL), m_size(0) {
#ifdef __VERBOSE__
            printf("creating empty hybrid pointer %x \n", this);
#endif
        }

        GT_FUNCTION
        explicit  hybrid_pointer(T* p, uint_t size_, bool externally_managed) : wrap_pointer<T>(p, size_, externally_managed), m_gpu_p(NULL), m_pointer_to_use(p), m_size(size_) {
	  allocate_it(m_size);
	}


        //GT_FUNCTION
        explicit hybrid_pointer(uint_t size, bool externally_managed=false) : wrap_pointer<T>(size, externally_managed), m_size(size), m_pointer_to_use (wrap_pointer<T>::m_cpu_p) {
            allocate_it(size);

#ifdef __VERBOSE__
            printf("allocating hybrid pointer %x \n", this);
            printf(" - %X %X %X %d\n", this->m_cpu_p, m_gpu_p, m_pointer_to_use, m_size);
#endif
    }

// copy constructor passes on the ownership
        GT_FUNCTION
        hybrid_pointer(hybrid_pointer const& other)
            : wrap_pointer<T>(other)
            , m_gpu_p(other.m_gpu_p)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 3200)
            , m_pointer_to_use(m_gpu_p)
#else
            , m_pointer_to_use(this->m_cpu_p)
#endif
            , m_size(other.m_size)
        {
#ifdef __VERBOSE__
            printf("cpy const hybrid pointer: ");
            printf("%X ", this->m_cpu_p);
            printf("%X ", m_gpu_p);
            printf("%X ", m_pointer_to_use);
            printf("%d ", m_size);
            printf("\n");
#endif
        }

        GT_FUNCTION
        virtual ~hybrid_pointer(){
#ifdef __VERBOSE__
            printf("deleting hybrid pointer %x \n", this);
#endif
};

        void allocate_it(uint_t size) {
#ifdef __CUDACC__
            cudaError_t err = cudaMalloc(&m_gpu_p, size*sizeof(T));
            if (err != cudaSuccess) {
                std::cout << "Error allocating storage in "
                          << BOOST_CURRENT_FUNCTION
                          << " : size = "
                          << size*sizeof(T)
                          << " bytes   " <<  cudaGetErrorString(err)
                          << std::endl;
#ifdef __VERBOSE__
                printf("allocating hybrid pointer %x \n", this);
#endif
            }
#endif
        }

        void free_it() {
#ifdef __CUDACC__
            cudaFree(m_gpu_p);
            m_gpu_p=NULL;
#endif
            wrap_pointer<T>::free_it();
#ifdef __VERBOSE__
            printf("freeing hybrid pointer %x \n", this);
#endif
      }

        void update_gpu() const {
#ifdef __CUDACC__
#ifdef __VERBOSE__
            printf("update gpu "); out();
#endif
            cudaMemcpy(m_gpu_p, this->m_cpu_p, m_size*sizeof(T), cudaMemcpyHostToDevice);
#endif
        }

        void update_cpu() const {
#ifdef __CUDACC__
#ifdef __VERBOSE__
            printf("update cpu "); out();
#endif
            cudaMemcpy(this->m_cpu_p, m_gpu_p, m_size*sizeof(T), cudaMemcpyDeviceToHost);
#endif
        }

#ifdef __CUDACC__
        void set(pointee_t const& value, uint_t const& index){cudaMemcpy(&m_pointer_to_use[index], &value, sizeof(pointee_t), cudaMemcpyHostToDevice); }
#endif

        __host__ __device__
        void out() const {
            printf("out hp ");
            printf("%X ", this->m_cpu_p);
            printf("%X ", m_gpu_p);
            printf("%X ", m_pointer_to_use);
            printf("%d ", m_size);
            printf("\n");
        }

        __host__ __device__
        operator T*() {
            return m_pointer_to_use;
        }

        __host__ __device__
        operator T const*() const {
            return m_pointer_to_use;
        }

        __host__ __device__
        T& operator[](uint_t i) {
            /* assert(i<size); */
            /* assert(i>=0); */
            // printf(" [%d %e] ", i, m_pointer_to_use[i]);
            return m_pointer_to_use[i];
        }

        __host__ __device__
        T const& operator[](uint_t i) const {
            /* assert(i<size); */
            /* assert(i>=0); */
            // printf(" [%d %e] ", i, m_pointer_to_use[i]);

            return m_pointer_to_use[i];
        }

        __host__ __device__
        T& operator*() {
            return *m_pointer_to_use;
        }

        __host__ __device__
        T const& operator*() const {
            return *m_pointer_to_use;
        }

        __host__ __device__
        T* operator+(uint_t i) {
            return &m_pointer_to_use[i];
        }

        __host__ __device__
        T* const& operator+(uint_t i) const {
            return &m_pointer_to_use[i];
        }

        GT_FUNCTION
        T* get_gpu_p(){return m_gpu_p;};

        GT_FUNCTION
        T* get_cpu_p(){return this->m_cpu_p;};

        GT_FUNCTION
        T* get_pointer_to_use(){return m_pointer_to_use;}

        GT_FUNCTION
        pointee_t* get() const {return m_gpu_p;}

        GT_FUNCTION
        int get_size(){return m_size;}

        /** the standard = operator */
        hybrid_pointer operator =(hybrid_pointer const& other){
            this->m_cpu_p = other.m_cpu_p;
            this->m_externally_managed = other.m_externally_managed;
            m_gpu_p = other.m_gpu_p;
            m_pointer_to_use =other.m_pointer_to_use;
            m_size = other.m_size;
        }
    private:
        /** disable equal operator and constructor from raw pointer*/
        T* operator =(T*);
        hybrid_pointer(T*);
        T * m_gpu_p;
        T * m_pointer_to_use;
        uint_t m_size;
    };

} // namespace gridtools
