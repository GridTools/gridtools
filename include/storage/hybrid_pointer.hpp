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
    struct hybrid_pointer // : public wrap_pointer<T>
    {

        // typedef wrap_pointer<T> super;
        typedef typename wrap_pointer<T>::pointee_t pointee_t;

        GT_FUNCTION
        explicit  hybrid_pointer() :
            m_gpu_p(NULL)
            , m_cpu_p((T*)NULL)
            , m_pointer_to_use(NULL)
            , m_size(0)
            , m_allocated(false)
            , m_up_to_date(true)
        {
#ifdef VERBOSE
            printf("creating empty hybrid pointer %x \n", this);
#endif
        }

        GT_FUNCTION
        explicit  hybrid_pointer(T* p, uint_t size_, bool externally_managed) :
            m_gpu_p(NULL)
            , m_cpu_p(p, size_, externally_managed)
            , m_pointer_to_use(p)
            , m_size(size_)
            , m_allocated(false)
            , m_up_to_date(true)
        {
            allocate_it(m_size);
        }


        //GT_FUNCTION
        explicit hybrid_pointer(uint_t size, bool externally_managed=false) :
            m_gpu_p(NULL)
            , m_cpu_p(size, externally_managed)
            , m_pointer_to_use (m_cpu_p.get()), m_size(size)
            , m_allocated(false)
            , m_up_to_date(true)
        {
            allocate_it(size);

#ifdef VERBOSE
            printf("allocating hybrid pointer %x \n", this);
            printf(" - %X %X %X %d\n", this->m_cpu_p, m_gpu_p, m_pointer_to_use, m_size);
#endif
    }

// copy constructor passes on the ownership
        GT_FUNCTION
        hybrid_pointer(hybrid_pointer const& other)
            :
            m_gpu_p(other.m_gpu_p)
            , m_cpu_p(other.m_cpu_p)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 3200)
            , m_pointer_to_use(m_gpu_p)
#else
            , m_pointer_to_use(this->m_cpu_p)
#endif
            , m_size(other.m_size)
            , m_allocated(other.m_allocated)
            , m_up_to_date(other.m_up_to_date)
        {
#ifdef VERBOSE
            printf("cpy const hybrid pointer: ");
            printf("%X ", m_cpu_p.get());
            printf("%X ", m_gpu_p);
            printf("%X ", m_pointer_to_use);
            printf("%d ", m_size);
            printf("\n");
#endif
        }


        GT_FUNCTION
        ~hybrid_pointer(){
#ifdef VERBOSE
            printf("deleting hybrid pointer %x \n", this);
#endif
};

        void allocate_it(uint_t size) {
            if(!m_allocated){
                cudaError_t err = cudaMalloc(&m_gpu_p, size*sizeof(T));
                m_up_to_date=false;
                if (err != cudaSuccess) {
                    std::cout << "Error allocating storage in "
                              << BOOST_CURRENT_FUNCTION
                              << " : size = "
                              << size*sizeof(T)
                              << " bytes   " <<  cudaGetErrorString(err)
                              << std::endl;
#ifdef VERBOSE
                    printf("allocating hybrid pointer %x \n", this);
#endif
                }
            }
        }

        void free_it() {
            cudaFree(m_gpu_p);
            m_gpu_p=NULL;
            m_cpu_p.free_it();
#ifdef VERBOSE
            printf("freeing hybrid pointer %x \n", this);
#endif
      }

        void update_gpu() const {
#ifdef VERBOSE
            printf("update gpu "); out();
#endif
            cudaMemcpy(m_gpu_p, m_cpu_p.get(), m_size*sizeof(T), cudaMemcpyHostToDevice);
            m_up_to_date=false;
        }

        void update_cpu() const {
#ifdef VERBOSE
            printf("update cpu "); out();
#endif
            if(!m_up_to_date){
                cudaMemcpy(m_cpu_p.get(), m_gpu_p, m_size*sizeof(T), cudaMemcpyDeviceToHost);
                m_up_to_date=true;
            }
        }

        void set(pointee_t const& value, uint_t const& index)
        {
            cudaMemcpy(&m_pointer_to_use[index], &value, sizeof(pointee_t), cudaMemcpyHostToDevice);
            m_up_to_date=false;
        }

        __host__ __device__
        void out() const {
            printf("out hp ");
            printf("%X ", m_cpu_p.get());
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
            // assert(m_pointer_to_use);
            // assert(i<m_size);
            // assert(i>=0);
            // printf(" [%d %e] ", i, m_pointer_to_use[i]);
            return m_pointer_to_use[i];
        }

        __host__ __device__
        T const& operator[](uint_t i) const {
            // assert(m_pointer_to_use);
            // assert(i<m_size);
            // assert(i>=0);
            // printf(" [%d %e] ", i, m_pointer_to_use[i]);

            return m_pointer_to_use[i];
        }

        __host__ __device__
        T& operator*() {
            // assert(m_pointer_to_use);
            return *m_pointer_to_use;
        }

        __host__ __device__
        T const& operator*() const {
            // assert(m_pointer_to_use);
            return *m_pointer_to_use;
        }

        __host__ __device__
        T* operator+(uint_t i) {
            // assert(m_pointer_to_use);
            return &m_pointer_to_use[i];
        }

        __host__ __device__
        T* const& operator+(uint_t i) const {
            // assert(m_pointer_to_use);
            return &m_pointer_to_use[i];
        }

        GT_FUNCTION
        T* get_gpu_p(){return m_gpu_p;};

        GT_FUNCTION
        T* get_cpu_p(){
            assert(on_host())
            return this->m_cpu_p.get();};

        GT_FUNCTION
        T* get_pointer_to_use(){return m_pointer_to_use;}

        GT_FUNCTION
        pointee_t* get() const {return m_gpu_p;}

        GT_FUNCTION
        int get_size(){return m_size;}

        GT_FUNCTION
        bool on_host(){
            return m_up_to_date;
        }

        GT_FUNCTION
        bool on_device(){
            return !m_up_to_date;
        }

        /**
           @brief swapping two pointers
         */
        GT_FUNCTION
        void swap(hybrid_pointer& other){
            m_cpu_p.swap(other.m_cpu_p);
            T* tmp = m_gpu_p;
            m_gpu_p = other.m_gpu_p;
            other.m_gpu_p = tmp;
            tmp = m_pointer_to_use;
            m_pointer_to_use = other.m_pointer_to_use;
            other.m_pointer_to_use = tmp;
            uint_t tmp_size = m_size;
            m_size = other.m_size;
            other.m_size = tmp_size;
        }

        void reset(T* cpu_p){m_cpu_p.reset(cpu_p);}

        bool set_externally_managed(bool externally_managed_){m_cpu_p.set_externally_managed(externally_managed_);}

        bool is_externally_managed() const {return m_cpu_p.is_externally_managed();}

        /** the standard = operator */
        hybrid_pointer operator =(hybrid_pointer const& other){
            m_gpu_p = other.m_gpu_p;
            m_cpu_p.reset(other.m_cpu_p.get());
            m_cpu_p.set_externally_managed(other.is_externally_managed());
            m_pointer_to_use =other.m_pointer_to_use;
            m_size = other.m_size;
        }
    private:
        /** disable equal operator and constructor from raw pointer*/
        T* operator =(T*);
        hybrid_pointer(T*);
        T * m_gpu_p;
        wrap_pointer<T> m_cpu_p;
        T * m_pointer_to_use;
        uint_t m_size;
        bool m_allocated;
        bool m_up_to_date;
    };

} // namespace gridtools
