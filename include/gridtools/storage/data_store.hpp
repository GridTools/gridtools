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

            decltype(auto) operator()(int offset) const { return m_fun(index_from_offset<Is>(offset)...); }
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
    class data_store {
        static_assert(is_storage<Storage>::value, GT_INTERNAL_ERROR_MSG("Passed type is no storage type"));
        static_assert(
            is_storage_info<StorageInfo>::value, GT_INTERNAL_ERROR_MSG("Passed type is no storage_info type"));
        static_assert(std::is_trivially_copyable<typename Storage::data_t>::value,
            "data_store only supports trivially copyable types.");

        struct impl {
            StorageInfo m_storage_info;
            Storage m_storage;
            std::string m_name;

            impl(StorageInfo const &info, std::string name)
                : m_storage_info(info), m_storage(info.padded_total_length(),
                                            info.first_index_of_inner_region(),
                                            typename StorageInfo::alignment_t()),
                  m_name(std::move(name)) {}

            impl(StorageInfo const &info, typename Storage::data_t *ptr, ownership own, std::string name)
                : m_storage_info(info), m_storage(info.padded_total_length(), ptr, own), m_name(std::move(name)) {}
        };

        std::shared_ptr<impl> m_impl;

        template <class Initializer>
        data_store(std::true_type, StorageInfo const &info, Initializer &&initializer, std::string name)
            : data_store(info, std::move(name)) {
            int length = info.padded_total_length();
            auto *dst = storage().get_cpu_ptr();
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int i = 0; i < length; ++i)
                dst[i] = initializer(i);
            storage().clone_to_device();
        }

      public:
        using data_t = typename Storage::data_t;
        using storage_info_t = StorageInfo;
        using storage_t = Storage;

        data_store(StorageInfo const &info) : data_store(info, "") {}

        // binary ctors

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * @param info storage_info instance
         * @param name Human readable name for the data_store
         */
        data_store(StorageInfo const &info, std::string name) : m_impl(std::make_shared<impl>(info, std::move(name))) {}

        data_store(StorageInfo const &info, data_t initializer) : data_store(info, initializer, "") {}

        // trinary ctors

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Additionally the data is initialized to the given value.
         * @param info storage info instance
         * @param initializer initialization value
         * @param name Human readable name for the data_store
         */
        data_store(StorageInfo const &info, data_t initializer, std::string name)
            : data_store(std::true_type(), info, [initializer](int) { return initializer; }, std::move(name)) {}

        template <class Initializer,
            std::enable_if_t<!std::is_convertible<Initializer &&, data_t>::value &&
                                 !std::is_convertible<Initializer &&, data_t *>::value &&
                                 !std::is_convertible<Initializer &&, std::string>::value,
                int> = 0>
        data_store(StorageInfo const &info, Initializer &&initializer)
            : data_store(info, std::forward<Initializer>(initializer), "") {}

        template <class T, std::enable_if_t<std::is_same<data_t *, T>::value, int> = 0>
        data_store(StorageInfo const &info, T ptr) : data_store(info, ptr, ownership::external_cpu, "") {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Additionally the data is initialized with the given value. Current i, j, k, etc. is passed
         * to the lambda.
         * @param info storage info instance
         * @param initializer initialization lambda
         * @param name Human readable name for the data_store
         */
        template <class Initializer,
            std::enable_if_t<!std::is_convertible<Initializer, data_t>::value &&
                                 !std::is_convertible<Initializer, data_t *>::value &&
                                 !std::is_convertible<Initializer, std::string>::value,
                int> = 0>
        data_store(StorageInfo const &info, Initializer &&initializer, std::string name)
            : data_store(std::true_type(),
                  info,
                  data_store_impl_::initializer_adapter_f<Initializer, StorageInfo>{initializer, info},
                  std::move(name)) {}

        data_store(StorageInfo const &info, data_t *ptr, ownership own) : data_store(info, ptr, own, "") {}

        // ctor from four args

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Either the host or the device pointer is external. This means the storage does not own
         * both sides. This is used when external data sources are used (e.g., Fortran or Python).
         * @param info storage info instance
         * @param external_ptr the external pointer
         * @param own ownership information
         * @param name Human readable name for the data_store
         */
        data_store(StorageInfo const &info, data_t *ptr, ownership own, std::string name)
            : m_impl(std::make_shared<impl>(info, ptr, own, std::move(name))) {}

        void swap(data_store &other) {
            using std::swap;
            swap(m_impl, other.m_impl);
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
            return info().template total_length<Dim>();
        }

        /**
         * @brief member function to retrieve the total size (dimensions, halos, padding).
         * @return total size
         */
        auto padded_total_length() const { return info().padded_total_length(); }

        /**
         * @brief member function to retrieve the inner domain size + halo (dimensions, halos).
         * @return inner domain size + halo
         */
        auto total_length() const { return info().total_length(); }

        /**
         * @brief member function to retrieve the inner domain size (dimensions, no halos).
         * @return inner domain size
         */
        auto length() const { return info().length(); }

        /**
         * @brief forward total_lengths() from storage_info
         */
        decltype(auto) total_lengths() const { return info().total_lengths(); }

        /**
         * @brief forward strides() from storage_info
         */
        decltype(auto) strides() const { return info().strides(); }

        /**
         * @brief retrieve the underlying storage_info instance
         * @return storage_info instance
         */
        StorageInfo const &info() const {
            assert(m_impl);
            return m_impl->m_storage_info;
        }

        /**
         * @brief retrieve a reference to the underlying storage instance.
         */
        Storage &storage() const {
            assert(m_impl);
            return m_impl->m_storage;
        }

        /**
         * @brief clone underlying storage to device
         */
        void clone_to_device() const { storage().clone_to_device(); }

        /**
         * @brief clone underlying storage from device
         */
        void clone_from_device() const { storage().clone_from_device(); }

        /**
         * @brief synchronize underlying storage
         */
        void sync() const { storage().sync(); }

        /**
         * @brief reactivate all device read write views to storage
         */
        void reactivate_target_write_views() const { storage().reactivate_target_write_views(); }

        /**
         * @brief reactivate all host read write views to storage
         */
        void reactivate_host_write_views() const { storage().reactivate_host_write_views(); }

        bool device_needs_update() const { return storage().device_needs_update_impl(); }

        bool host_needs_update() const { return storage().host_needs_update_impl(); }

        /**
         * @brief retrieve the name of the storage
         * @return name of the data_store
         */
        std::string const &name() const {
            assert(m_impl);
            return m_impl->m_name;
        }

        friend bool operator==(const data_store &lhs, const data_store &rhs) { return lhs.m_impl == rhs.m_impl; }
        friend bool operator!=(const data_store &lhs, const data_store &rhs) { return !(lhs == rhs); }
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
