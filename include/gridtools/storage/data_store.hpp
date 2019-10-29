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
#include "../common/layout_map.hpp"
#include "../meta/type_traits.hpp"
#include "common/alignment.hpp"
#include "common/definitions.hpp"
#include "common/halo.hpp"
#include "common/storage_info.hpp"
#include "common/storage_interface.hpp"

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

            decltype(auto) operator()(int i) const {
                auto indices = m_info.indices(i);
                return m_fun(indices[Is]...);
            }
        };
    } // namespace data_store_impl_

    /** \ingroup storage
     * @brief data_store implementation. This struct wraps storage and storage information in one class.
     * It can be copied and passed around without replicating the data. Automatic cleanup is provided when
     * the last data_store that points to the data is destroyed.
     * @tparam Storage storage type that should be used (e.g., cuda_storage)
     * @tparam StorageInfo storage info type that should be used (e.g., cuda_storage_info)
     */
    template <class Storage, class StorageInfo>
    class data_store;

    template <typename Storage, uint_t Id, int... LayoutArgs, uint_t... Halos, uint_t Align, class Indices>
    class data_store<Storage, storage_info<Id, layout_map<LayoutArgs...>, halo<Halos...>, alignment<Align>, Indices>> {
      public:
        using storage_info_t = storage_info<Id, layout_map<LayoutArgs...>, halo<Halos...>, alignment<Align>, Indices>;

      private:
        static_assert(is_storage<Storage>::value, GT_INTERNAL_ERROR_MSG("Passed type is no storage type"));
        static_assert(std::is_trivially_copyable<typename Storage::data_t>::value,
            "data_store only supports trivially copyable types.");

        struct impl {
            storage_info_t m_storage_info;
            Storage m_storage;
            std::string m_name;

            impl(storage_info_t const &info, std::string name)
                : m_storage_info(info), m_storage(info.length(), info.index(Halos...), alignment<Align>()),
                  m_name(std::move(name)) {}

            impl(storage_info_t const &info, typename Storage::data_t *ptr, ownership own, std::string name)
                : m_storage_info(info), m_storage(info.length(), ptr, own), m_name(std::move(name)) {}
        };

        std::shared_ptr<impl> m_impl;

        template <class Initializer>
        data_store(std::true_type, storage_info_t const &info, Initializer &&initializer, std::string name)
            : data_store(info, std::move(name)) {
            int length = info.length();
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
        using storage_t = Storage;

        data_store(storage_info_t const &info) : data_store(info, "") {}

        // binary ctors

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * @param info storage_info instance
         * @param name Human readable name for the data_store
         */
        data_store(storage_info_t const &info, std::string name)
            : m_impl(std::make_shared<impl>(info, std::move(name))) {}

        data_store(storage_info_t const &info, data_t initializer) : data_store(info, initializer, "") {}

        template <class Initializer,
            std::enable_if_t<!std::is_convertible<Initializer &&, data_t>::value &&
                                 !std::is_convertible<Initializer &&, data_t *>::value &&
                                 !std::is_convertible<Initializer &&, std::string>::value,
                int> = 0>
        data_store(storage_info_t const &info, Initializer &&initializer)
            : data_store(info, std::forward<Initializer>(initializer), "") {}

        template <class T, std::enable_if_t<std::is_same<data_t *, T>::value, int> = 0>
        data_store(storage_info_t const &info, T ptr) : data_store(info, ptr, ownership::external_cpu, "") {}

        // trinary ctors

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Additionally the data is initialized to the given value.
         * @param info storage info instance
         * @param initializer initialization value
         * @param name Human readable name for the data_store
         */
        data_store(storage_info_t const &info, data_t initializer, std::string name)
            : data_store(std::true_type(), info, [initializer](int) { return initializer; }, std::move(name)) {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Additionally the data is initialized with the given value. Current i, j, k, etc. is passed
         * to the lambda.
         * @param info storage info instance
         * @param initializer initialization lambda
         * @param name Human readable name for the data_store
         */
        template <class Initializer, std::enable_if_t<!std::is_convertible<Initializer, data_t>::value, int> = 0>
        data_store(storage_info_t const &info, Initializer &&initializer, std::string name)
            : data_store(std::true_type(),
                  info,
                  data_store_impl_::initializer_adapter_f<Initializer, storage_info_t>{initializer, info},
                  std::move(name)) {}

        data_store(storage_info_t const &info, data_t *ptr, ownership own) : data_store(info, ptr, own, "") {}

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
        data_store(storage_info_t const &info, data_t *ptr, ownership own, std::string name)
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
        auto length() const {
            return info().template length<Dim>();
        }

        /**
         * @brief member function to retrieve the total size (dimensions, halos, padding).
         * @return total size
         */
        auto length() const { return info().length(); }

        /**
         * @brief forward total_lengths() from storage_info
         */
        decltype(auto) lengths() const { return info().lengths(); }

        /**
         * @brief forward strides() from storage_info
         */
        decltype(auto) strides() const { return info().strides(); }

        /**
         * @brief retrieve the underlying storage_info instance
         * @return storage_info instance
         */
        storage_info_t const &info() const {
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
