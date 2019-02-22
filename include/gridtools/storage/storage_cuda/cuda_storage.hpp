/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <array>
#include <utility>
#include <vector>

#include "../../common/cuda_util.hpp"
#include "../../common/gt_assert.hpp"
#include "../common/state_machine.hpp"
#include "../common/storage_interface.hpp"
#include "../storage_host/host_storage.hpp"

namespace gridtools {
    /** \ingroup storage
     * @{
     */

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
    template <typename DataType>
    struct cuda_storage : storage_interface<cuda_storage<DataType>> {
        typedef DataType data_t;
        typedef std::array<DataType *, 2> ptrs_t;
        typedef state_machine state_machine_t;

      private:
        cuda_util::unique_cuda_ptr<DataType> m_gpu_ptr_holder;
        std::unique_ptr<DataType[]> m_cpu_ptr_holder;

        DataType *m_gpu_ptr;
        DataType *m_cpu_ptr;
        state_machine m_state;
        uint_t m_size;

      public:
        /*
         * @brief cuda_storage constructor. Just allocates enough memory on Host and Device.
         * @param size defines the size of the storage and the allocated space.
         */
        template <uint_t Align = 1>
        cuda_storage(uint_t size, uint_t offset_to_align = 0u, alignment<Align> = alignment<1u>{})
            : m_gpu_ptr_holder(cuda_util::cuda_malloc<DataType>(size + Align - 1)),
              m_cpu_ptr_holder(new DataType[size]), m_cpu_ptr(m_cpu_ptr_holder.get()), m_state{}, m_size{size} {
            DataType *allocated_ptr = m_gpu_ptr_holder.get();
            auto delta =
                (reinterpret_cast<std::uintptr_t>(allocated_ptr + offset_to_align) % (Align * sizeof(DataType))) /
                sizeof(DataType);
            m_gpu_ptr = delta == 0 ? allocated_ptr : allocated_ptr + Align - delta;
        }

        /*
         * @brief cuda_storage constructor. Does not allocate memory on both sides but uses one external pointer.
         * Reason for having this is to support externally allocated memory (e.g., from Fortran or Python).
         * Allocates memory either on Host or Device.
         * @param size defines the size of the storage and the allocated space.
         * @param external_ptr a pointer to the external data
         * @param own ownership information (external CPU pointer, or external GPU pointer)
         */
        explicit cuda_storage(uint_t size, DataType *external_ptr, ownership own)
            : m_gpu_ptr_holder(own != ownership::external_gpu ? cuda_util::cuda_malloc<DataType>(size)
                                                              : cuda_util::unique_cuda_ptr<DataType>()),
              m_cpu_ptr_holder(own == ownership::external_cpu ? nullptr : new DataType[size]),
              m_gpu_ptr(own == ownership::external_gpu ? external_ptr : m_gpu_ptr_holder.get()),
              m_cpu_ptr(own == ownership::external_cpu ? external_ptr : m_cpu_ptr_holder.get()),
              m_state{own != ownership::external_cpu, own != ownership::external_gpu}, m_size{size} {
            assert(external_ptr);
        }

        /*
         * @brief cuda_storage constructor. Allocates memory on Host and Device and initializes the memory according to
         * the given initializer.
         * @param size defines the size of the storage and the allocated space.
         * @param initializer initialization value
         */
        template <typename Fun, uint_t Align = 1>
        cuda_storage(uint_t size, Fun &&initializer, uint_t offset_to_align = 0u, alignment<Align> a = alignment<1u>{})
            : cuda_storage(size, offset_to_align, a) {
            for (uint_t i = 0; i < size; ++i)
                m_cpu_ptr[i] = initializer(i);
            this->clone_to_device();
        }

        /*
         * @brief swap implementation for cuda_storage
         */
        void swap_impl(cuda_storage &other) {
            using std::swap;
            swap(m_gpu_ptr_holder, other.m_gpu_ptr_holder);
            swap(m_cpu_ptr_holder, other.m_cpu_ptr_holder);
            swap(m_gpu_ptr, other.m_gpu_ptr);
            swap(m_cpu_ptr, other.m_cpu_ptr);
            swap(m_state, other.m_state);
            swap(m_size, other.m_size);
        }

        /*
         * @brief retrieve the device data pointer.
         * @return device pointer
         */
        DataType *get_gpu_ptr() const {
            GT_ASSERT_OR_THROW(m_gpu_ptr, "This storage has never been initialized.");
            return m_gpu_ptr;
        }

        /*
         * @brief retrieve the host data pointer.
         * @return host pointer
         */
        DataType *get_cpu_ptr() const {
            GT_ASSERT_OR_THROW(m_cpu_ptr, "This storage has never been initialized.");
            return m_cpu_ptr;
        }

        /*
         * @brief clone_to_device implementation for cuda_storage.
         */
        void clone_to_device_impl() {
            GT_ASSERT_OR_THROW(m_cpu_ptr, "CPU pointer seems not initialized.");
            GT_ASSERT_OR_THROW(m_gpu_ptr, "GPU pointer seems not initialized.");

            GT_CUDA_CHECK(cudaMemcpy(m_gpu_ptr, m_cpu_ptr, m_size * sizeof(DataType), cudaMemcpyHostToDevice));
            m_state = {};
        }

        /*
         * @brief clone_from_device implementation for cuda_storage.
         */
        void clone_from_device_impl() {
            GT_CUDA_CHECK(cudaMemcpy(m_cpu_ptr, m_gpu_ptr, m_size * sizeof(DataType), cudaMemcpyDeviceToHost));
            m_state = {};
        }

        /*
         * @brief synchronization implementation for cuda_storage.
         */
        void sync_impl() {
            // check if we can avoid syncing (in case neither host or device needs an update)
            if (!m_state.m_hnu && !m_state.m_dnu)
                return;
            // invalid state occurs when both host and device would need an update.
            GT_ASSERT_OR_THROW((m_state.m_hnu ^ m_state.m_dnu), "invalid state detected.");
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
            GT_ASSERT_OR_THROW(!m_state.m_dnu, "host views are in write mode");
            m_state.m_hnu = 1;
        }

        /*
         * @brief reactivate_host_write_views implementation for cuda_storage.
         */
        void reactivate_host_write_views_impl() {
            GT_ASSERT_OR_THROW(!m_state.m_hnu, "device views are in write mode");
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
         * @brief valid implementation for cuda_storage.
         */
        bool valid_impl() const { return m_cpu_ptr && m_gpu_ptr; }
    };

    // simple metafunction to check if a type is a cuda storage
    template <typename T>
    struct is_cuda_storage : std::false_type {};

    template <typename T>
    struct is_cuda_storage<cuda_storage<T>> : std::true_type {};

    /**
     * @}
     */
} // namespace gridtools
