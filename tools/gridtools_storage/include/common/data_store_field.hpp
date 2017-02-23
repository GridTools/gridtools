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

#include <boost/mpl/vector_c.hpp>

#include "data_store.hpp"
#include "data_store_field_metafunctions.hpp"
#include "storage_info_interface.hpp"

namespace gridtools {

    template < typename DataStore, unsigned... N >
    struct data_store_field {
        using data_store_t = DataStore;
        using data_t = typename DataStore::data_t;
        using storage_t = typename DataStore::storage_t;
        using state_machine_t = typename DataStore::state_machine_t;
        using storage_info_t = typename DataStore::storage_info_t;

        const static unsigned size = get_accumulated_data_field_index(sizeof...(N), N...);
        const static unsigned dims = sizeof...(N);

        // tuple of arrays (e.g., { {s00,s01,s02}, {s10, s11}, {s20} }, 3-dimensional field with snapshot sizes 3, 2,
        // and 1. All together we have 6 storages.)
        std::array< DataStore, size > m_field;
        constexpr data_store_field(storage_info_t const &info) : m_field(emplace_array< DataStore, size >(info)) {}

        // same as above but with different storage infos (given per component)
        template < typename... StorageInfos >
        data_store_field(StorageInfos const &... infos)
            : m_field(get_vals< typename get_sequence< boost::mpl::vector<>,
                      N... >::type >::template generator< data_store_t, storage_info_t >(infos...)) {
            static_assert(sizeof...(StorageInfos) == dims, "Only one storage info per component allowed.");
        }

        template < unsigned Dim, unsigned Snapshot >
        DataStore &get() {
            return m_field[get_accumulated_data_field_index(Dim, N...) + Snapshot];
        }

        DataStore &get(unsigned Dim, unsigned Snapshot) {
            return m_field[get_accumulated_data_field_index(Dim, N...) + Snapshot];
        }

        template < unsigned Dim, unsigned Snapshot >
        void set(DataStore &store) {
            m_field[get_accumulated_data_field_index(Dim, N...) + Snapshot] = store;
        }

        void set(unsigned Dim, unsigned Snapshot, DataStore &store) {
            m_field[get_accumulated_data_field_index(Dim, N...) + Snapshot] = store;
        }

        void allocate() {
            for (auto &e : m_field)
                e.allocate();
        }

        void free() {
            for (auto &e : m_field)
                e.free();
        }

        bool valid() const {
            bool res = true;
            for (auto &e : m_field)
                res &= e.valid();
            return res;
        }

        decltype(m_field) const &get_field() const { return m_field; }

        constexpr std::array< unsigned, sizeof...(N) > get_dim_sizes() const { return {N...}; }

        // forwarding methods
        void clone_to_device() const {
            for (auto &e : this->m_field)
                e.clone_to_device();
        }

        void clone_from_device() const {
            for (auto &e : this->m_field)
                e.clone_from_device();
        }

        void sync() const {
            for (auto &e : this->m_field)
                e.sync();
        }

        void reactivate_device_write_views() const {
            for (auto &e : this->m_field)
                e.reactivate_device_write_views();
        }

        void reactivate_host_write_views() const {
            for (auto &e : this->m_field)
                e.reactivate_host_write_views();
        }
    };

    // simple metafunction to check if a type is a data_store_field
    template < typename T, unsigned... N >
    struct is_data_store_field : boost::mpl::false_ {};

    template < typename S, unsigned... N >
    struct is_data_store_field< data_store_field< S, N... > > : boost::mpl::true_ {};

    /**
     *  Implementation of a swap function. E.g., swap<0,0>::with<0,1>(field_view)
     *  will swap the CPU and GPU pointers of storages 0,0 and 0,1.
     *  This operation invalidates the previously created views.
     **/
    template < unsigned Dim_S, unsigned Snapshot_S >
    struct swap {
        template < unsigned Dim_T, unsigned Snapshot_T, typename T, unsigned... N >
        static void with(data_store_field< T, N... > &data_field) {
            typedef typename std::remove_pointer< decltype(
                std::declval< typename data_store_field< T, N... >::data_store_t >().get_storage_ptr()) >::type::ptrs_t
                ptrs_t;
            auto &src = data_field.template get< Dim_S, Snapshot_S >();
            auto &trg = data_field.template get< Dim_T, Snapshot_T >();
            auto tmp_cpu = src.get_storage_ptr()->template get_ptrs< ptrs_t >();
            src.get_storage_ptr()->template set_ptrs< ptrs_t >(trg.get_storage_ptr()->template get_ptrs< ptrs_t >());
            trg.get_storage_ptr()->template set_ptrs< ptrs_t >(tmp_cpu);
        }
    };
}
