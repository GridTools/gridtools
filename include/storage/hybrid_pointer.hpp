/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "wrap_pointer.hpp"

/** @file
    @brief double pointer mapping host and device
    implementation of a double pointer, living on both host and device, together with the algorithms to copy back/to the
   device. The device is supposed to be a GPU supporting CUDA.
*/
namespace gridtools {

    /**\todo Note that this struct will greatly simplify when the CUDA arch 3200 and inferior will be obsolete (the
     * "pointer_to_use" will then become useless, and the operators defined in the base class will be usable) */
    template < typename T, bool Array = true >
    struct hybrid_pointer {
      private:
        template < typename V >
        hybrid_pointer(V);

      public:
        // typedef wrap_pointer<T> super;
        typedef typename wrap_pointer< T, Array >::pointee_t pointee_t;

        GT_FUNCTION
        explicit hybrid_pointer()
            : m_gpu_p(NULL), m_cpu_p(static_cast< T * >(NULL), false), m_pointer_to_use(NULL), m_size(0),
              m_allocated(false), m_up_to_date(true) {
#ifdef VERBOSE
            printf("creating empty hybrid pointer %x \n", this);
#endif
        }

        GT_FUNCTION
        explicit hybrid_pointer(T *p, uint_t size_, bool externally_managed = false)
            : m_gpu_p(NULL), m_cpu_p(p, externally_managed), m_pointer_to_use(p), m_size(size_), m_allocated(false),
              m_up_to_date(true) {
            allocate_it(m_size);
        }

        explicit hybrid_pointer(T *p, bool externally_managed = false)
            : m_gpu_p(NULL), m_cpu_p(p, externally_managed), m_pointer_to_use(p), m_size(1), m_allocated(false),
              m_up_to_date(true) {
            allocate_it(m_size);
        }

        // GT_FUNCTION
        explicit hybrid_pointer(uint_t size, bool externally_managed = false)
            : m_gpu_p(NULL), m_cpu_p(size, externally_managed), m_pointer_to_use(m_cpu_p.get()), m_size(size),
              m_allocated(false), m_up_to_date(true) {
            allocate_it(size);

#ifdef VERBOSE
            printf("allocating hybrid pointer %x \n", this);
            printf(" - %X %X %X %d\n", m_cpu_p.get(), m_gpu_p, m_pointer_to_use, m_size);
#endif
        }

        // copy constructor passes on the ownership
        GT_FUNCTION
        hybrid_pointer(hybrid_pointer const &other)
            : m_gpu_p(other.m_gpu_p), m_cpu_p(other.m_cpu_p)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 3200)
              ,
              m_pointer_to_use(m_gpu_p)
#else
              ,
              m_pointer_to_use(this->m_cpu_p.get())
#endif
              ,
              m_size(other.m_size), m_allocated(other.m_allocated), m_up_to_date(other.m_up_to_date) {
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
        ~hybrid_pointer() {
#ifdef VERBOSE
            printf("deleting hybrid pointer %x \n", this);
#endif
        };

        void allocate_it(uint_t size) {
            if (!m_allocated) {
                cudaError_t err = cudaMalloc(&m_gpu_p, size * sizeof(T));
                m_allocated = true;
                if (err != cudaSuccess) {
                    std::cout << "Error allocating storage in "
                              // << BOOST_CURRENT_FUNCTION
                              << " : size = " << size * sizeof(T) << " bytes   " << cudaGetErrorString(err)
                              << std::endl;
#ifdef VERBOSE
                    printf("allocating hybrid pointer %x \n", this);
#endif
                }
            }
        }

        void free_it() {
            if (m_gpu_p) // if the pointers are not allocated do nothing
            {
                assert(m_gpu_p); // check for double free
                cudaError_t err = cudaFree((void *)(m_gpu_p));
                assert(err == cudaSuccess);
                m_gpu_p = NULL;
                m_cpu_p.free_it();
                m_up_to_date = true;
                m_pointer_to_use = NULL;
                m_allocated = false;

#ifdef VERBOSE
                printf("freeing hybrid pointer %x \n", this);
#endif
            }
        }

        void update_gpu() {
#ifdef VERBOSE
            printf("update gpu ");
            out();
#endif
            if (on_host()) { // do not copy if the last version is already on the device
                cudaError_t err =
                    cudaMemcpy((void *)m_gpu_p, (void *)m_cpu_p.get(), m_size * sizeof(T), cudaMemcpyHostToDevice);
                assert(err == cudaSuccess);
                m_up_to_date = false;
                m_pointer_to_use = m_gpu_p;
            }
        }

        void update_cpu() {
#ifdef VERBOSE
            printf("update cpu ");
            out();
#endif
            if (on_device()) {
                cudaError_t err =
                    cudaMemcpy((void *)m_cpu_p.get(), (void *)m_gpu_p, m_size * sizeof(T), cudaMemcpyDeviceToHost);
                assert(err == cudaSuccess);
                m_up_to_date = true;
                m_pointer_to_use = m_cpu_p.get();
            }
        }

        void set(pointee_t const &value, uint_t const &index) {
            if (on_host()) { // do not copy if the last version is already on the device
                cudaError_t err =
                    cudaMemcpy(&m_pointer_to_use[index], &value, sizeof(pointee_t), cudaMemcpyHostToDevice);
#ifndef __CUDACC__
                assert(err == cudaSuccess);
#endif
                m_up_to_date = false;
                m_pointer_to_use = m_gpu_p;
            }
        }

        GT_FUNCTION
        void out() const {
            printf("out hp ");
            printf("%X ", m_cpu_p.get());
            printf("%X ", m_gpu_p);
            printf("%X ", m_pointer_to_use);
            printf("%d ", m_size);
            printf("\n");
        }

        GT_FUNCTION
        operator T *() { return m_pointer_to_use; }

        GT_FUNCTION
        operator T const *() const { return m_pointer_to_use; }

        GT_FUNCTION
        T &operator[](uint_t i) {
            // assert(m_pointer_to_use);
            // assert(i<m_size);
            // assert(i>=0);
            // printf(" [%d %e] ", i, m_pointer_to_use[i]);
            return m_pointer_to_use[i];
        }

        GT_FUNCTION
        T const &operator[](uint_t i) const {
            // assert(m_pointer_to_use);
            // assert(i<m_size);
            // assert(i>=0);
            // printf(" [%d %e] ", i, m_pointer_to_use[i]);

            return m_pointer_to_use[i];
        }

        /**
           @brief access operator
         */
        GT_FUNCTION
        T *operator->() const {
#ifndef __CUDACC__
            assert(m_pointer_to_use);
#endif
            return m_pointer_to_use;
        }

        GT_FUNCTION
        T &operator*() {
            // assert(m_pointer_to_use);
            return *m_pointer_to_use;
        }

        GT_FUNCTION
        T const &operator*() const {
            // assert(m_pointer_to_use);
            return *m_pointer_to_use;
        }

        GT_FUNCTION
        T *operator+(uint_t i) {
            // assert(m_pointer_to_use);
            return &m_pointer_to_use[i];
        }

        GT_FUNCTION
        T *const &operator+(uint_t i) const {
            // assert(m_pointer_to_use);
            return &m_pointer_to_use[i];
        }

        GT_FUNCTION
        T *get_gpu_p() const {
#ifndef __CUDACC__
            assert(on_device());
#endif
            return m_gpu_p;
        };

        GT_FUNCTION
        T *get_cpu_p() const {
#ifndef __CUDACC__
            assert(on_host());
#endif
            return m_cpu_p.get();
        };

        GT_FUNCTION
        T *get_pointer_to_use() { return m_pointer_to_use; }

        GT_FUNCTION
        T *get_pointer_to_use() const { return m_pointer_to_use; }

        GT_FUNCTION
        pointee_t *get() const { return m_gpu_p; }

        GT_FUNCTION
        int get_size() { return m_size; }

        GT_FUNCTION
        void set_on_device() {
            m_up_to_date = false;
            m_pointer_to_use = m_gpu_p;
        }

        GT_FUNCTION
        void set_on_host() {
            m_up_to_date = true;
            m_pointer_to_use = m_cpu_p;
        }

        GT_FUNCTION
        bool on_host() const { return m_up_to_date; }

        GT_FUNCTION
        bool on_device() const { return !m_up_to_date; }

        /**
           @brief swapping two pointers
         */
        GT_FUNCTION
        void swap(hybrid_pointer &other) {
            m_cpu_p.swap(other.m_cpu_p);

            {
                T *tmp = m_gpu_p;
                m_gpu_p = other.m_gpu_p;
                other.m_gpu_p = tmp;
            }

            {
                T *tmp2 = m_pointer_to_use;
                m_pointer_to_use = other.m_pointer_to_use;
                other.m_pointer_to_use = tmp2;
            }

            {
                uint_t tmp_size = m_size;
                m_size = other.m_size;
                other.m_size = tmp_size;
            }

            {
                bool tmp_allocated = m_allocated;
                m_allocated = other.m_allocated;
                other.m_allocated = tmp_allocated;
            }

            {
                bool tmp_up_to_date = m_up_to_date;
                m_up_to_date = other.m_up_to_date;
                other.m_up_to_date = tmp_up_to_date;
            }
        }

        GT_FUNCTION
        void reset(T *cpu_p) { m_cpu_p.reset(cpu_p); }

        GT_FUNCTION
        void set_externally_managed(bool externally_managed_) { m_cpu_p.set_externally_managed(externally_managed_); }

        GT_FUNCTION
        bool is_externally_managed() const { return m_cpu_p.is_externally_managed(); }

        /** the standard = operator */
        GT_FUNCTION
        hybrid_pointer &operator=(hybrid_pointer const &other) {
            m_gpu_p = other.m_gpu_p;
            m_cpu_p = other.m_cpu_p;
            m_pointer_to_use = other.m_pointer_to_use;
            m_size = other.m_size;
            m_allocated = other.m_allocated;
            m_up_to_date = other.m_up_to_date;
            return *this;
        }

      private:
        /** disable equal operator and constructor from raw pointer*/
        T *operator=(T *);
        hybrid_pointer(T *);
        T *m_gpu_p;
        wrap_pointer< T, Array > m_cpu_p;
        T *m_pointer_to_use;
        uint_t m_size;
        bool m_allocated;
        bool m_up_to_date;
    };

} // namespace gridtools
