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

#include <array>
#include <assert.h>

#include <boost/mpl/bool.hpp>

#include "../../common/gt_assert.hpp"
#include "../common/state_machine.hpp"
#include "../common/storage_interface.hpp"
#include "../storage_host/host_storage.hpp"

namespace gridtools {

    /*
     * @brief The CUDA storage implementation. This class owns the CPU and GPU pointer
     * to the data. Additionally there is a state machine that keeps information about
     * the current state and a field that knows about size and ownership. Instances of
     * this class are noncopyable.
     * @tparam DataType the type of the data and the pointers respectively (e.g., float or double)
     *
     * Here we are using the CRTP. Actually the same
     * functionality could be implemented using standard inheritance
     * but we prefer the CRTP because it can be seen as the standard
     * gridtools pattern and we clearly want to avoid virtual
     * methods, etc.
     */
    template < typename DataType >
    struct cuda_storage : storage_interface< cuda_storage< DataType > > {
        typedef DataType data_t;
        typedef std::array< data_t *, 2 > ptrs_t;
        typedef state_machine state_machine_t;

      private:
        data_t *m_gpu_ptr;
        data_t *m_cpu_ptr;
        state_machine m_state;
        uint_t m_size;
        ownership m_ownership = ownership::Full;

      public:
        /*
         * @brief cuda_storage constructor. Just allocates enough memory on Host and Device.
         * @param size defines the size of the storage and the allocated space.
         */
        cuda_storage(uint_t size) : m_cpu_ptr(new data_t[size]), m_size(size) {
            cudaError_t err = cudaMalloc(&m_gpu_ptr, size * sizeof(data_t));
            ASSERT_OR_THROW((err == cudaSuccess), "failed to allocate GPU memory.");
        }

        /*
         * @brief cuda_storage constructor. Does not allocate memory on both sides but uses one external pointer.
         * Reason for having this is to support externally allocated memory (e.g., from Fortran or Python).
         * Allocates memory either on Host or Device.
         * @param size defines the size of the storage and the allocated space.
         * @param external_ptr a pointer to the external data
         * @param own ownership information (external CPU pointer, or external GPU pointer)
         */
        explicit cuda_storage(uint_t size, data_t *external_ptr, ownership own) : m_size(size), m_ownership(own) {
            ASSERT_OR_THROW(((own == ownership::ExternalGPU) || (own == ownership::ExternalCPU)),
                "external pointer cuda_storage ownership must be either ExternalGPU or ExternalCPU.");
            if (own == ownership::ExternalGPU) {
                m_cpu_ptr = new data_t[size];
                m_gpu_ptr = external_ptr;
                m_state.m_hnu = true;
            } else if (own == ownership::ExternalCPU) {
                m_cpu_ptr = external_ptr;
                cudaError_t err = cudaMalloc(&m_gpu_ptr, size * sizeof(data_t));
                ASSERT_OR_THROW((err == cudaSuccess), "failed to allocate GPU memory.");
                m_state.m_dnu = true;
            }
            ASSERT_OR_THROW((m_gpu_ptr && m_cpu_ptr), "Failed to create cuda_storage.");
        }

        /*
         * @brief cuda_storage constructor. Allocates memory on Host and Device and initializes the memory according to
         * the given initializer.
         * @param size defines the size of the storage and the allocated space.
         * @param initializer initialization value
         */
        cuda_storage(uint_t size, data_t initializer) : m_cpu_ptr(new data_t[size]), m_size(size) {
            for (uint_t i = 0; i < size; ++i) {
                m_cpu_ptr[i] = initializer;
            }
            cudaError_t err = cudaMalloc(&m_gpu_ptr, size * sizeof(data_t));
            ASSERT_OR_THROW((err == cudaSuccess), "failed to allocate GPU memory.");
            this->clone_to_device();
        }

        /*
         * @brief cuda_storage destructor.
         */
        ~cuda_storage() {
            if ((m_ownership == ownership::ExternalGPU || m_ownership == ownership::Full) && m_cpu_ptr)
                delete[] m_cpu_ptr;
            if ((m_ownership == ownership::ExternalCPU || m_ownership == ownership::Full) && m_gpu_ptr)
                cudaFree(m_gpu_ptr);
        }

        /*
         * @brief retrieve the device data pointer.
         * @return device pointer
         */
        data_t *get_gpu_ptr() const {
            ASSERT_OR_THROW(m_gpu_ptr, "This storage has never been initialized.");
            return m_gpu_ptr;
        }

        /*
         * @brief retrieve the host data pointer.
         * @return host pointer
         */
        data_t *get_cpu_ptr() const {
            ASSERT_OR_THROW(m_cpu_ptr, "This storage has never been initialized.");
            return m_cpu_ptr;
        }

        /*
         * @brief clone_to_device implementation for cuda_storage.
         */
        void clone_to_device_impl() {
            cudaError_t err =
                cudaMemcpy((void *)m_gpu_ptr, (void *)this->m_cpu_ptr, m_size * sizeof(data_t), cudaMemcpyHostToDevice);
            ASSERT_OR_THROW((err == cudaSuccess), "failed to clone data to the device.");
            m_state.m_hnu = false;
            m_state.m_dnu = false;
        }

        /*
         * @brief clone_from_device implementation for cuda_storage.
         */
        void clone_from_device_impl() {
            cudaError_t err =
                cudaMemcpy((void *)this->m_cpu_ptr, (void *)m_gpu_ptr, m_size * sizeof(data_t), cudaMemcpyDeviceToHost);
            ASSERT_OR_THROW((err == cudaSuccess), "failed to clone data from the device.");
            m_state.m_hnu = false;
            m_state.m_dnu = false;
        }

        /*
         * @brief synchronization implementation for cuda_storage.
         */
        void sync_impl() {
            // check if we can avoid syncing (in case neither host or device needs an update)
            if (!m_state.m_hnu && !m_state.m_dnu)
                return;
            // invalid state occurs when both host and device would need an update.
            ASSERT_OR_THROW((m_state.m_hnu ^ m_state.m_dnu), "invalid state detected.");
            // sync
            if (m_state.m_hnu) { // if host needs update clone the data from the device
                this->clone_from_device();
            } else if (m_state.m_dnu) { // if device needs update clone the data to the device
                this->clone_to_device();
            }
        }

        /*
         * @brief device_needs_update implementation for cuda_storage.
         */
        bool device_needs_update_impl() const { return m_state.m_dnu; }

        /*
         * @brief host_needs_update implementation for cuda_storage.
         */
        bool host_needs_update_impl() const { return m_state.m_hnu; }

        /*
         * @brief reactivate_device_write_views implementation for cuda_storage.
         */
        void reactivate_device_write_views_impl() {
            ASSERT_OR_THROW(!m_state.m_dnu, "host views are in write mode");
            m_state.m_hnu = 1;
        }

        /*
         * @brief reactivate_host_write_views implementation for cuda_storage.
         */
        void reactivate_host_write_views_impl() {
            ASSERT_OR_THROW(!m_state.m_hnu, "device views are in write mode");
            m_state.m_dnu = 1;
        }

        /*
         * @brief get_state_machine_ptr implementation for cuda_storage.
         */
        state_machine *get_state_machine_ptr_impl() { return &m_state; }

        /*
         * @brief get_ptrs implementation for cuda_storage.
         */
        ptrs_t get_ptrs_impl() const { return {m_cpu_ptr, m_gpu_ptr}; }

        /*
         * @brief set_ptrs implementation for cuda_storage.
         */
        void set_ptrs_impl(ptrs_t const &ptrs) {
            m_gpu_ptr = ptrs[1];
            m_cpu_ptr = ptrs[0];
        }

        /*
         * @brief valid implementation for cuda_storage.
         */
        bool valid_impl() const { return m_cpu_ptr && m_gpu_ptr; }
    };

    // simple metafunction to check if a type is a cuda storage
    template < typename T >
    struct is_cuda_storage : boost::mpl::false_ {};

    template < typename T >
    struct is_cuda_storage< cuda_storage< T > > : boost::mpl::true_ {};
}
