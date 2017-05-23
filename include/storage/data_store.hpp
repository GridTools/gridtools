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
#include <array>
#include <string>

#include <boost/mpl/bool.hpp>

#include "common/definitions.hpp"
#include "common/storage_interface.hpp"
#include "common/storage_info_interface.hpp"

namespace gridtools {

    namespace {
        /**
         * @brief metafunction used to retrieve the appropriate function type that is needed in order
         * to be able to initialize the data_store (underlying storage) with a given value (step case).
         * E.g., It is not possible to initialize a 2 dimensional container with a lambda that expects 3 arguments.
         * @tparam ReturnType the return type of the function or lambda
         * @tparam StorageInfo storage_info type
         * @tparam N the number of dimensions (e.g., layout_map<2,1,0> -> 3 dimensions)
         * @tparam Args variadic pack of int types
         */
        template < typename ReturnType,
            typename StorageInfo,
            unsigned N = StorageInfo::layout_t::masked_length,
            typename... Args >
        struct appropriate_function_t {
            typedef typename appropriate_function_t< ReturnType, StorageInfo, N - 1, Args..., int >::type type;
        };

        /**
         * @brief metafunction used to retrieve the appropriate function type that is needed in order
         * to be able to initialize the data_store (underlying storage) with a given value (base case).
         * E.g., It is not possible to initialize a 2 dimensional container with a lambda that expects 3 arguments.
         * @tparam ReturnType the return type of the function or lambda
         * @tparam StorageInfo storage_info type
         * @tparam Args variadic pack of int types
         */
        template < typename ReturnType, typename StorageInfo, typename... Args >
        struct appropriate_function_t< ReturnType, StorageInfo, 0, Args... > {
            typedef std::function< ReturnType(Args...) > type;
        };

        /**
         * @brief helper function used to initialize a storage with a given lambda (base case).
         * The reason for having this is that generic initializations should be supported.
         * E.g., a 4-dimensional storage should be initialize-able with a lambda of
         * type data_t(int, int, int, int).
         * @tparam Lambda the lambda type
         * @tparam StorageInfo storage_info type
         * @tparam DataType value type of the container
         * @tparam Args variadic list of integers
         * @param init lambda instance
         * @param si storage info object
         * @param ptr storage cpu pointer
         * @param args pack that contains the current index for each dimension
         */
        template < typename Lambda, typename StorageInfo, typename DataType, typename... Args >
        typename boost::enable_if_c< (sizeof...(Args) == StorageInfo::layout_t::masked_length - 1), void >::type
        lambda_initializer(Lambda init, StorageInfo si, DataType *ptr, Args... args) {
            for (unsigned i = 0; i < si.template unaligned_dim< sizeof...(Args) >(); ++i) {
                ptr[si.index(args..., i)] = init(args..., i);
            }
        }

        /**
         * @brief helper function used to initialize a storage with a given lambda (step case).
         * The reason for having this is that generic initializations should be supported.
         * E.g., a 4-dimensional storage should be initialize-able with a lambda of
         * type data_t(int, int, int, int).
         * @tparam Lambda the lambda type
         * @tparam StorageInfo storage_info type
         * @tparam DataType value type of the container
         * @tparam Args variadic list of integers
         * @param init lambda instance
         * @param si storage info object
         * @param ptr storage cpu pointer
         * @param args pack that contains the current index for each dimension
         */
        template < typename Lambda, typename StorageInfo, typename DataType, typename... Args >
        typename boost::enable_if_c< (sizeof...(Args) < StorageInfo::layout_t::masked_length - 1), void >::type
        lambda_initializer(Lambda init, StorageInfo si, DataType *ptr, Args... args) {
            for (unsigned i = 0; i < si.template unaligned_dim< sizeof...(Args) >(); ++i) {
                lambda_initializer(init, si, ptr, args..., i);
            }
        }
    }

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
        std::string m_name;

      public:
        /**
         * @brief data_store constructor. This constructor does not trigger an allocation of the required space.
         */
        constexpr data_store(std::string const &name = "")
            : m_shared_storage(nullptr), m_shared_storage_info(nullptr), m_name(name) {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * @param info storage info instance
         */
        constexpr data_store(StorageInfo const &info, std::string const &name = "")
            : m_shared_storage(new storage_t(info.size())), m_shared_storage_info(new storage_info_t(info)),
              m_name(name) {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Additionally the data is initialized to the given value.
         * @param info storage info instance
         * @param initializer initialization value
         */
        constexpr data_store(StorageInfo const &info, data_t initializer, std::string const &name = "")
            : m_shared_storage(new storage_t(info.size(), initializer)),
              m_shared_storage_info(new storage_info_t(info)), m_name(name) {}

        /**
         * @brief data_store constructor. This constructor triggers an allocation of the required space.
         * Additionally the data is initialized with the given value. Current i, j, k, etc. is passed
         * to the lambda.
         * @param info storage info instance
         * @param initializer initialization lambda
         */
        data_store(StorageInfo const &info,
            typename appropriate_function_t< data_t, StorageInfo >::type const &initializer,
            std::string const &name = "")
            : m_shared_storage(new storage_t(info.size())), m_shared_storage_info(new storage_info_t(info)),
              m_name(name) {
            // initialize the storage with the given lambda
            lambda_initializer(initializer, info, m_shared_storage->get_cpu_ptr());
            // synchronize contents
            clone_to_device();
        }

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
        explicit constexpr data_store(StorageInfo const &info,
            T external_ptr,
            ownership own = ownership::ExternalCPU,
            std::string const &name = "")
            : m_shared_storage(new storage_t(info.size(), external_ptr, own)),
              m_shared_storage_info(new storage_info_t(info)), m_name(name) {}

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
            : m_shared_storage(other.m_shared_storage), m_shared_storage_info(other.m_shared_storage_info),
              m_name(other.m_name) {
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
            m_name = other.m_name;
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

        /*
         * @brief function to retrieve the (aligned) size of a dimension (e.g., I, J, or K).
         * @tparam Coord queried coordinate
         * @return size of dimension
         */
        template < int Coord >
        int dim() const {
            assert(m_shared_storage_info.get() && "data_store is in a non-initialized state.");
            return m_shared_storage_info->template dim< Coord >();
        }

        /*
         * @brief member function to retrieve the total size (dimensions, halos, initial_offset).
         * @return total size
         */
        int size() const {
            assert(m_shared_storage_info.get() && "data_store is in a non-initialized state.");
            return m_shared_storage_info->size();
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

        bool device_needs_update() const { return this->m_shared_storage->device_needs_update_impl(); }

        bool host_needs_update() const { return this->m_shared_storage->host_needs_update_impl(); }

        /**
         * @brief retrieve the name of the storage
         * @return name of the data_store
         */
        std::string const &name() const { return m_name; }
    };

    // simple metafunction to check if a type is a cuda_data_store
    template < typename T >
    struct is_data_store : boost::mpl::false_ {};

    template < typename S, typename SI >
    struct is_data_store< data_store< S, SI > > : boost::mpl::true_ {};
}
