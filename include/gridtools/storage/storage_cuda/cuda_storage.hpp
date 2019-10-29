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
#include "../common/storage_interface.hpp"
#include "../data_view.hpp"
#include "../storage_host/host_storage.hpp"
#include "cuda_storage_info.hpp"
#include "state_machine.hpp"

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

    namespace cuda_storage_impl_ {
        template <typename DataType>
        class cuda_storage : public storage_interface<cuda_storage<DataType>> {
          private:
            cuda_util::unique_cuda_ptr<DataType> m_gpu_ptr_holder;
            std::unique_ptr<DataType[]> m_cpu_ptr_holder;

            DataType *m_gpu_ptr;
            DataType *m_cpu_ptr;
            state_machine m_state;
            uint_t m_size;

          public:
            using data_t = DataType;

            /*
             * @brief cuda_storage constructor. Just allocates enough memory on Host and Device.
             * @param size defines the size of the storage and the allocated space.
             */
            template <uint_t Align>
            cuda_storage(uint_t size, uint_t offset_to_align, alignment<Align>)
                : m_gpu_ptr_holder(cuda_util::cuda_malloc<DataType>(size + Align - 1)),
                  m_cpu_ptr_holder(new DataType[size]), m_cpu_ptr(m_cpu_ptr_holder.get()), m_size{size} {
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
            cuda_storage(uint_t size, DataType *external_ptr, ownership own)
                : m_gpu_ptr_holder(own != ownership::external_gpu ? cuda_util::cuda_malloc<DataType>(size)
                                                                  : cuda_util::unique_cuda_ptr<DataType>()),
                  m_cpu_ptr_holder(own == ownership::external_cpu ? nullptr : new DataType[size]),
                  m_gpu_ptr(own == ownership::external_gpu ? external_ptr : m_gpu_ptr_holder.get()),
                  m_cpu_ptr(own == ownership::external_cpu ? external_ptr : m_cpu_ptr_holder.get()), m_size(size) {
                if (own == ownership::external_cpu)
                    m_state.touch_host();
                else
                    m_state.touch_device();
                assert(external_ptr);
            }

            /*
             * @brief retrieve the device data pointer.
             * @return device pointer
             */
            DataType *get_target_ptr_impl() const { return m_gpu_ptr; }

            /*
             * @brief retrieve the host data pointer.
             * @return host pointer
             */
            DataType *get_cpu_ptr_impl() const { return m_cpu_ptr; }

            /*
             * @brief clone_to_device implementation for cuda_storage.
             */
            void clone_to_device_impl() {
                GT_CUDA_CHECK(cudaMemcpy(m_gpu_ptr, m_cpu_ptr, m_size * sizeof(DataType), cudaMemcpyHostToDevice));
                m_state = {};
            }

            /*
             * @brief clone_from_device implementation for cuda_storage.
             */
            void clone_from_device_impl() {
                if (m_state.host_needs_update())
                    GT_CUDA_CHECK(cudaMemcpy(m_cpu_ptr, m_gpu_ptr, m_size * sizeof(DataType), cudaMemcpyDeviceToHost));
                m_state = {};
            }

            /*
             * @brief synchronization implementation for cuda_storage.
             */
            void sync_impl() {
                if (m_state.host_needs_update())
                    clone_from_device_impl();
                else if (m_state.device_needs_update())
                    clone_to_device_impl();
            }

            /*
             * @brief device_needs_update implementation for cuda_storage.
             */
            bool device_needs_update_impl() const { return m_state.device_needs_update(); }

            /*
             * @brief host_needs_update implementation for cuda_storage.
             */
            bool host_needs_update_impl() const { return m_state.host_needs_update(); }

            /*
             * @brief reactivate_target_write_views implementation for cuda_storage.
             */
            void reactivate_target_write_views_impl() { m_state.touch_device(); }

            /*
             * @brief reactivate_host_write_views implementation for cuda_storage.
             */
            void reactivate_host_write_views_impl() { m_state.touch_host(); }

            state_machine &state() { return m_state; }
            state_machine const &state() const { return m_state; }
        };

        template <class T, class Mode, class Type, class Info>
        host_view<typename Type::type, Info> make_host_view_impl(
            Mode mode, Type, cuda_storage<T> const &storage, Info const &info) {
            storage.state().touch_host(mode);
            return {storage.get_cpu_ptr(), &info};
        }

        template <class T, class Info, class U>
        bool check_consistency_impl(cuda_storage<T> const &storage, host_view<U, Info> const &view) {
            return view.data() == storage.get_cpu_ptr() && storage.state().device_needs_update();
        }

        template <class T, class Info, class U>
        bool check_consistency_impl(cuda_storage<T> const &storage, host_view<U const, Info> const &view) {
            return view.data() == storage.get_cpu_ptr() && !storage.state().host_needs_update();
        }

        template <class T, class StorageInfo>
        struct device_view {
            T *m_ptr;
            StorageInfo const *m_info;

            using storage_info_t = StorageInfo;
            using data_t = T;

            GT_FUNCTION_DEVICE StorageInfo const &info() const { return *m_info; }

            GT_FUNCTION_DEVICE T *data() const { return m_ptr; }

            template <class... Coords>
            GT_FUNCTION_DEVICE T &operator()(Coords... c) const {
                return m_ptr[m_info->index(c...)];
            }

            GT_FUNCTION_DEVICE T &operator()(array<int, StorageInfo::ndims> const &arr) const {
                return m_ptr[m_info->index(arr)];
            }

            GT_FUNCTION_DEVICE GT_CONSTEXPR auto length() const { return m_info->length(); }

            GT_FUNCTION_DEVICE GT_CONSTEXPR decltype(auto) lengths() const { return m_info->lengths(); }

            friend T *advanced_get_raw_pointer_of(device_view const &src) { return src.m_ptr; }
        };

        template <class T, class StorageInfo, class U>
        bool check_consistency_impl(cuda_storage<T> &storage, device_view<U, StorageInfo> const &view) {
            return advanced_get_raw_pointer_of(view) == storage.get_target_ptr() && storage.state().host_needs_update();
        }

        template <class T, class StorageInfo, class U>
        bool check_consistency_impl(cuda_storage<T> &storage, device_view<U const, StorageInfo> const &view) {
            return advanced_get_raw_pointer_of(view) == storage.get_target_ptr() &&
                   !storage.state().device_needs_update();
        }

        template <class T, class Mode, class Type, class Info>
        device_view<typename Type::type, Info> make_target_view_impl(
            Mode mode, Type, cuda_storage<T> &storage, Info const &info) {
            storage.state().touch_device(mode);
            return {storage.get_target_ptr(), get_gpu_storage_info_ptr(info)};
        }
    } // namespace cuda_storage_impl_

    using cuda_storage_impl_::cuda_storage;

    /**
     * @}
     */
} // namespace gridtools
