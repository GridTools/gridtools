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

#include <array>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <boost/mpl/bool.hpp>

#include "../storage_host/storage.hpp"
#include "../common/storage_interface.hpp"
#include "../common/state_machine.hpp"

namespace gridtools {

    template < typename T >
    struct cuda_storage : storage_interface< cuda_storage< T > > {
        typedef T data_t;
        typedef std::array< T *, 2 > ptrs_t;
        typedef state_machine state_machine_t;

      private:
        T *m_gpu_ptr;
        T *m_cpu_ptr;
        state_machine m_state;
        unsigned m_size;

      public:
        cuda_storage(unsigned size) : m_cpu_ptr(new T[size]), m_size(size) {

            cudaError_t err = cudaMalloc(&m_gpu_ptr, size * sizeof(T));
            assert((err == cudaSuccess) && "failed to allocate GPU memory.");
        }
        ~cuda_storage() {
            assert(m_gpu_ptr && "This would end up in a double-free.");
            cudaFree(m_gpu_ptr);
            assert(m_cpu_ptr && "This would end up in a double-free.");
            delete[] m_cpu_ptr;
        }

        T *get_gpu_ptr() const {
            assert(m_gpu_ptr && "This storage has never been initialized.");
            return m_gpu_ptr;
        }

        T *get_cpu_ptr() const {
            assert(m_cpu_ptr && "This storage has never been initialized.");
            return m_cpu_ptr;
        }

        // cloning methods
        void clone_to_device_impl() {
            cudaError_t err =
                cudaMemcpy((void *)m_gpu_ptr, (void *)this->m_cpu_ptr, m_size * sizeof(T), cudaMemcpyHostToDevice);
            assert((err == cudaSuccess) && "failed to clone data to the device.");
            m_state.m_od = true;
            m_state.m_hnu = false;
            m_state.m_dnu = false;
        }

        void clone_from_device_impl() {
            cudaError_t err =
                cudaMemcpy((void *)this->m_cpu_ptr, (void *)m_gpu_ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost);
            assert((err == cudaSuccess) && "failed to clone data from the device.");
            m_state.m_od = false;
            m_state.m_hnu = false;
            m_state.m_dnu = false;
        }

        void sync_impl() {
            if (!m_state.m_hnu && !m_state.m_dnu)
                return;
            assert((m_state.m_hnu ^ m_state.m_dnu) && "invalid state detected.");
            (m_state.m_hnu) ? this->clone_from_device() : this->clone_to_device();
        }

        // checking the state
        bool is_on_host_impl() const { return !m_state.m_od; }
        bool is_on_device_impl() const { return m_state.m_od; }
        bool device_needs_update_impl() const { return m_state.m_dnu; }
        bool host_needs_update_impl() const { return m_state.m_hnu; }
        void reactivate_device_write_views_impl() {
            assert(!m_state.m_dnu && "host views are in write mode");
            m_state.m_hnu = 1;
            m_state.m_od = 1;
        }
        void reactivate_host_write_views_impl() {
            assert(!m_state.m_hnu && "device views are in write mode");
            m_state.m_dnu = 1;
            m_state.m_od = 0;
        }
        state_machine *get_state_machine_ptr_impl() { return &m_state; }

        // interface used for swap operations
        ptrs_t get_ptrs_impl() const { return {m_cpu_ptr, m_gpu_ptr}; }

        void set_ptrs_impl(ptrs_t const &ptrs) {
            m_gpu_ptr = ptrs[1];
            m_cpu_ptr = ptrs[0];
        }

        // interface to check validity
        bool valid_impl() const { return m_cpu_ptr && m_gpu_ptr; }
    };

    // simple metafunction to check if a type is a cuda storage
    template < typename T >
    struct is_cuda_storage : boost::mpl::false_ {};

    template < typename T >
    struct is_cuda_storage< cuda_storage< T > > : boost::mpl::true_ {};
}
