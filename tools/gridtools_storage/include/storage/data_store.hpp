/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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

#include <assert.h>
#include <memory>

#include <boost/mpl/bool.hpp>

#include "common/storage_interface.hpp"
#include "common/storage_info_interface.hpp"

namespace gridtools {

    /**
     * @brief data_store implementation. This struct wraps storage and storage information in one class.
     * It can be copied and passed around without replicating the data. Automatic cleanup is provided when
     * the last data_store that points to the data is destroyed.
     * @tparam Storage storage type that should be used (e.g., cuda_storage)
     * @tparam StorageInfo storage info type that should be used (e.g., cuda_storage_info)
     */
    template < typename Storage, typename StorageInfo >
    struct data_store {
        static_assert(is_storage< Storage >::value, "Passed type is no storage type");
        static_assert(is_storage_info< StorageInfo >::value, "Passed type is no storage_info type");
        typedef typename Storage::data_t data_t;
        typedef typename Storage::state_machine_t state_machine_t;
        typedef StorageInfo storage_info_t;
        typedef Storage storage_t;

      protected:
        std::shared_ptr< storage_t > m_shared_storage;
        std::shared_ptr< storage_info_t > m_shared_storage_info;

      public:
        /**
         * @brief data_store constructor. This constructor does not trigger an allocation of the required space.
         */
        constexpr data_store() : m_shared_storage(nullptr), m_shared_storage_info(nullptr) {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * @param info storage info instance
         */
        constexpr data_store(StorageInfo const &info)
            : m_shared_storage(new storage_t(info.size())), m_shared_storage_info(new storage_info_t(info)) {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Additionally the data is initialized to the given value.
         * @param info storage info instance
         * @param initializer initialization value
         */
        constexpr data_store(StorageInfo const &info, data_t initializer)
            : m_shared_storage(new storage_t(info.size(), initializer)),
              m_shared_storage_info(new storage_info_t(info)) {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Either the host or the device pointer is external. This means the storage does not own
         * both sides. This is used when external data sources are used (e.g., Fortran or Python).
         * @param info storage info instance
         * @param external_ptr the external pointer
         * @param own ownership information
         */
        template < typename T = data_t *,
            typename boost::enable_if_c< boost::is_pointer< T >::value && boost::is_same< data_t *, T >::value,
                int >::type = 0 >
        explicit constexpr data_store(StorageInfo const &info, T external_ptr, ownership own = ownership::ExternalCPU)
            : m_shared_storage(new storage_t(info.size(), external_ptr, own)),
              m_shared_storage_info(new storage_info_t(info)) {}

        /**
         * @brief data_store move constructor
         * @param other the copied object
         */
        data_store(data_store &&other) = default;

        /**
         * @brief data_store copy constructor
         * @param other the copied object
         */
        data_store(data_store const &other)
            : m_shared_storage(other.m_shared_storage), m_shared_storage_info(other.m_shared_storage_info) {
            assert(other.valid() && "Cannot copy a non-initialized data_store.");
        }

        /**
         * @brief copy assign operation.
         * @param other the rhs of the assignment
         */
        data_store &operator=(data_store const &other) {
            // check that the other storage is valid
            assert(other.valid() && "Cannot copy a non-initialized data_store.");
            // check that dimensions are compatible; in case the storage has not been
            // initialized yet, we don't check compatibility of the storage infos.
            assert(!valid() ||
                   (*m_shared_storage_info == *other.m_shared_storage_info) &&
                       "Cannot copy-assign a data store with incompatible storage info.");
            // copy the contents
            m_shared_storage = other.m_shared_storage;
            m_shared_storage_info = other.m_shared_storage_info;
            return *this;
        }

        /**
         * @brief allocate the needed memory. this will instantiate a storage instance.
         */
        void allocate(StorageInfo const &info) {
            assert((!m_shared_storage_info.get() && !m_shared_storage.get()) &&
                   "This data store has already been allocated.");
            m_shared_storage_info = std::make_shared< storage_info_t >(info);
            m_shared_storage = std::make_shared< storage_t >(m_shared_storage_info->size());
        }

        /**
         * @brief reset the data_store.
         */
        void reset() {
            m_shared_storage_info.reset();
            m_shared_storage.reset();
        }

        /**
         * @brief retrieve a pointer to the underlying storage instance.
         * @return pointer to the underlying storage instance
         */
        storage_t *get_storage_ptr() const {
            assert(m_shared_storage.get() && "data_store is in a non-initialized state.");
            return m_shared_storage.get();
        }

        /**
         * @brief retrieve a pointer to the underlying storage_info instance.
         * @return pointer to the underlying storage_info instance
         */
        storage_info_t const *get_storage_info_ptr() const {
            assert(m_shared_storage_info.get() && "data_store is in a non-initialized state.");
            return m_shared_storage_info.get();
        }

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
        void reactivate_device_write_views() const { this->m_shared_storage->reactivate_device_write_views(); }

        /**
         * @brief reactivate all host read write views to storage
         */
        void reactivate_host_write_views() const { this->m_shared_storage->reactivate_host_write_views(); }
    };

    // simple metafunction to check if a type is a cuda_data_store
    template < typename T >
    struct is_data_store : boost::mpl::false_ {};

    template < typename S, typename SI >
    struct is_data_store< data_store< S, SI > > : boost::mpl::true_ {};
}
