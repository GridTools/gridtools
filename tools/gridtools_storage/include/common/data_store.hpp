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

#include <assert.h>
#include <memory>

#include <boost/mpl/bool.hpp>

#include "storage_interface.hpp"
#include "storage_info_interface.hpp"

namespace gridtools {

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
        constexpr data_store(StorageInfo const &info)
            : m_shared_storage(nullptr), m_shared_storage_info(new storage_info_t(info)) {}

        constexpr data_store(StorageInfo const &info, data_t initializer)
            : m_shared_storage(new storage_t(info.size(), initializer)),
              m_shared_storage_info(new storage_info_t(info)) {}

        template < typename T = data_t *,
            typename boost::enable_if_c< boost::is_pointer< T >::value && boost::is_same< data_t *, T >::value,
                int >::type = 0 >
        explicit constexpr data_store(
            StorageInfo const &info, T external_ptr, enumtype::ownership ownership = enumtype::ExternalCPU)
            : m_shared_storage(new storage_t(info.size(), external_ptr, ownership)),
              m_shared_storage_info(new storage_info_t(info)) {}

        data_store(data_store &&other) = default;

        data_store(data_store const &other)
            : m_shared_storage(other.m_shared_storage), m_shared_storage_info(other.m_shared_storage_info) {
            assert(m_shared_storage.get() && "Cannot copy a non-initialized data_store.");
        }

        data_store &operator=(data_store const &other) {
            m_shared_storage = other.m_shared_storage;
            m_shared_storage_info = other.m_shared_storage_info;
            assert(m_shared_storage.get() && "Cannot copy a non-initialized data_store.");
            return *this;
        }

        void allocate() { m_shared_storage = std::make_shared< storage_t >(m_shared_storage_info->size()); }

        void free() { m_shared_storage.reset(); }

        storage_t *get_storage_ptr() const {
            assert(m_shared_storage.get() && "data_store is in a non-initialized state.");
            return m_shared_storage.get();
        }

        storage_info_t const *get_storage_info_ptr() const { return m_shared_storage_info.get(); }

        bool valid() const {
            return m_shared_storage.get() && m_shared_storage->valid() && m_shared_storage_info.get();
        }

        // forwarding methods
        void clone_to_device() const { this->m_shared_storage->clone_to_device(); }
        void clone_from_device() const { this->m_shared_storage->clone_from_device(); }
        void sync() const { this->m_shared_storage->sync(); }

        bool is_on_host() const { return this->m_shared_storage->is_on_host(); }
        bool is_on_device() const { return this->m_shared_storage->is_on_device(); }
        void reactivate_device_write_views() const { this->m_shared_storage->reactivate_device_write_views(); }
        void reactivate_host_write_views() const { this->m_shared_storage->reactivate_host_write_views(); }
    };

    // simple metafunction to check if a type is a cuda_data_store
    template < typename T >
    struct is_data_store : boost::mpl::false_ {};

    template < typename S, typename SI >
    struct is_data_store< data_store< S, SI > > : boost::mpl::true_ {};
}
