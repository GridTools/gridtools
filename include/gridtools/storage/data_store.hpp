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
#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "../common/gt_assert.hpp"
#include "../meta/type_traits.hpp"
#include "./common/definitions.hpp"
#include "./common/storage_info.hpp"
#include "./common/storage_interface.hpp"

namespace gridtools {

    /** \ingroup storage
     * @{
     */

    namespace data_store_impl_ {
        template <class Fun, class StorageInfo, class = std::make_index_sequence<StorageInfo::ndims>>
        struct initializer_adapter_f;

        template <class Fun, class StorageInfo, size_t... Is>
        struct initializer_adapter_f<Fun, StorageInfo, std::index_sequence<Is...>> {
            Fun const &m_fun;
            StorageInfo const &m_info;

            template <size_t Dim>
            int index_from_offset(int offset) const {
                auto stride = m_info.template stride<Dim>();
                return stride ? offset / stride % m_info.template padded_length<Dim>()
                              : m_info.template total_length<Dim>() - 1;
            }

            auto operator()(int offset) const { return m_fun(index_from_offset<Is>(offset)...); }
        };
    } // namespace data_store_impl_

    /** \ingroup storage
     * @brief data_store implementation. This struct wraps storage and storage information in one class.
     * It can be copied and passed around without replicating the data. Automatic cleanup is provided when
     * the last data_store that points to the data is destroyed.
     * @tparam Storage storage type that should be used (e.g., cuda_storage)
     * @tparam StorageInfo storage info type that should be used (e.g., cuda_storage_info)
     */
    template <typename Storage, typename StorageInfo>
    struct data_store {
        GT_STATIC_ASSERT(is_storage<Storage>::value, GT_INTERNAL_ERROR_MSG("Passed type is no storage type"));
        GT_STATIC_ASSERT(
            is_storage_info<StorageInfo>::value, GT_INTERNAL_ERROR_MSG("Passed type is no storage_info type"));
        typedef typename Storage::data_t data_t;
        typedef typename Storage::state_machine_t state_machine_t;
        typedef StorageInfo storage_info_t;
        typedef Storage storage_t;

      protected:
        std::shared_ptr<storage_t> m_shared_storage;
        std::shared_ptr<storage_info_t> m_shared_storage_info;
        std::string m_name;

      private:
        template <class Initializer>
        data_store(std::true_type, StorageInfo const &info, Initializer &&initializer, std::string const &name)
            : data_store(info, name) {
            int length = m_shared_storage_info->padded_total_length();
            auto *dst = m_shared_storage->get_cpu_ptr();
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int i = 0; i < length; ++i)
                dst[i] = initializer(i);
            m_shared_storage->clone_to_device();
        }

      public:
        /**
         * @brief data_store constructor. This constructor does not trigger an allocation of the required space.
         */
        data_store(std::string const &name = "") : m_name(name) {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * @param info storage_info instance
         * @param name Human readable name for the data_store
         */
        data_store(StorageInfo const &info, std::string const &name = "")
            : m_shared_storage(new storage_t(
                  info.padded_total_length(), info.first_index_of_inner_region(), typename StorageInfo::alignment_t{})),
              m_shared_storage_info(new storage_info_t(info)), m_name(name) {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Additionally the data is initialized to the given value.
         * @param info storage info instance
         * @param initializer initialization value
         * @param name Human readable name for the data_store
         */
        data_store(StorageInfo const &info, data_t const &initializer, std::string const &name = "")
            : data_store(std::true_type{}, info, [&initializer](int) { return initializer; }, name) {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Additionally the data is initialized with the given value. Current i, j, k, etc. is passed
         * to the lambda.
         * @param info storage info instance
         * @param initializer initialization lambda
         * @param name Human readable name for the data_store
         */
        template <class Initializer,
            std::enable_if_t<!std::is_convertible<Initializer, data_t const &>::value &&
                                 !std::is_convertible<Initializer, data_t *>::value &&
                                 !std::is_convertible<Initializer, std::string const &>::value,
                int> = 0>
        data_store(StorageInfo const &info, Initializer &&initializer, std::string const &name = "")
            : data_store(std::true_type{},
                  info,
                  data_store_impl_::initializer_adapter_f<Initializer, StorageInfo>{initializer, info},
                  name) {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Either the host or the device pointer is external. This means the storage does not own
         * both sides. This is used when external data sources are used (e.g., Fortran or Python).
         * @param info storage info instance
         * @param external_ptr the external pointer
         * @param own ownership information
         * @param name Human readable name for the data_store
         */
        template <class T, std::enable_if_t<std::is_same<data_t *, T>::value, int> = 0>
        data_store(StorageInfo const &info,
            T external_ptr,
            ownership own = ownership::external_cpu,
            std::string const &name = "")
            : m_shared_storage(new storage_t(info.padded_total_length(), external_ptr, own)),
              m_shared_storage_info(new storage_info_t(info)), m_name(name) {}

        /**
         * @brief allocate the needed memory. this will instantiate a storage instance.
         *
         * @param info StorageInfo instance
         */
        void allocate(StorageInfo const &info) {
            GT_ASSERT_OR_THROW(
                !m_shared_storage_info.get() && !m_shared_storage.get(), "This data store has already been allocated.");
            m_shared_storage_info = std::make_shared<storage_info_t>(info);
            m_shared_storage = std::make_shared<storage_t>(m_shared_storage_info->padded_total_length(),
                m_shared_storage_info->first_index_of_inner_region(),
                typename StorageInfo::alignment_t{});
        }

        /**
         * @brief reset the data_store.
         */
        void reset() {
            m_shared_storage_info.reset();
            m_shared_storage.reset();
        }

        void swap(data_store &other) {
            using std::swap;
            swap(m_shared_storage, other.m_shared_storage);
            swap(m_shared_storage_info, other.m_shared_storage_info);
            swap(m_name, other.m_name);
        }

        friend void swap(data_store &a, data_store &b) { a.swap(b); };

        /**
         * @brief function to retrieve the size of a dimension (e.g., I, J, or K).
         *
         * @tparam Coord queried coordinate
         * @return size of dimension (including halos but not padding)
         */
        template <int Dim>
        auto total_length() const {
            GT_ASSERT_OR_THROW((m_shared_storage_info.get()), "data_store is in a non-initialized state.");
            return m_shared_storage_info->template total_length<Dim>();
        }

        /**
         * @brief member function to retrieve the total size (dimensions, halos, padding).
         * @return total size
         */
        auto padded_total_length() const {
            GT_ASSERT_OR_THROW((m_shared_storage_info.get()), "data_store is in a non-initialized state.");
            return m_shared_storage_info->padded_total_length();
        }

        /**
         * @brief member function to retrieve the inner domain size + halo (dimensions, halos).
         * @return inner domain size + halo
         */
        auto total_length() const {
            GT_ASSERT_OR_THROW((m_shared_storage_info.get()), "data_store is in a non-initialized state.");
            return m_shared_storage_info->total_length();
        }

        /**
         * @brief member function to retrieve the inner domain size (dimensions, no halos).
         * @return inner domain size
         */
        auto length() const {
            GT_ASSERT_OR_THROW((m_shared_storage_info.get()), "data_store is in a non-initialized state.");
            return m_shared_storage_info->length();
        }

        /**
         * @brief forward total_lengths() from storage_info
         */
        decltype(auto) total_lengths() const {
            GT_ASSERT_OR_THROW((m_shared_storage_info.get()), "data_store is in a non-initialized state.");
            return m_shared_storage_info->total_lengths();
        }

        /**
         * @brief forward strides() from storage_info
         */
        decltype(auto) strides() const { return m_shared_storage_info->strides(); }

        /**
         * @brief retrieve the underlying storage_info instance
         * @return storage_info instance
         */
        storage_info_t const &info() const { return *m_shared_storage_info; }

        /**
         * @brief retrieve a pointer to the underlying storage instance.
         * @return shared pointer to the underlying storage instance
         */
        std::shared_ptr<storage_t> const &get_storage_ptr() const { return m_shared_storage; }

        /**
         * @brief retrieve a pointer to the underlying storage_info instance.
         * @return shared pointer to the underlying storage_info instance
         */
        std::shared_ptr<storage_info_t> get_storage_info_ptr() const { return m_shared_storage_info; }

        /**
         * @brief check if underlying storage info and storage is valid.
         * @return true if underlying elements are valid, false otherwise
         */
        bool valid() const {
            return m_shared_storage.get() && m_shared_storage->valid() && m_shared_storage_info.get();
        }

        /**
         * @brief clone underlying storage to device
         */
        void clone_to_device() const { this->m_shared_storage->clone_to_device(); }

        /**
         * @brief clone underlying storage from device
         */
        void clone_from_device() const { this->m_shared_storage->clone_from_device(); }

        /**
         * @brief synchronize underlying storage
         */
        void sync() const { this->m_shared_storage->sync(); }

        /**
         * @brief reactivate all device read write views to storage
         */
        void reactivate_target_write_views() const { this->m_shared_storage->reactivate_target_write_views(); }

        /**
         * @brief reactivate all host read write views to storage
         */
        void reactivate_host_write_views() const { this->m_shared_storage->reactivate_host_write_views(); }

        bool device_needs_update() const { return this->m_shared_storage->device_needs_update_impl(); }

        bool host_needs_update() const { return this->m_shared_storage->host_needs_update_impl(); }

        /**
         * @brief retrieve the name of the storage
         * @return name of the data_store
         */
        std::string const &name() const { return m_name; }

        friend bool operator==(const data_store &lhs, const data_store &rhs) {
            return std::tie(lhs.m_name, lhs.m_shared_storage, lhs.m_shared_storage_info) ==
                   std::tie(rhs.m_name, rhs.m_shared_storage, rhs.m_shared_storage_info);
        }
        friend bool operator!=(const data_store &lhs, const data_store &rhs) { return !(lhs == rhs); }

        explicit operator bool() const { return valid(); }
    }; // namespace gridtools

    /// @brief simple metafunction to check if a type is a data_store
    template <typename T>
    struct is_data_store : std::false_type {};

    template <typename S, typename SI>
    struct is_data_store<data_store<S, SI>> : std::true_type {};

    /**
     * @}
     */
} // namespace gridtools
