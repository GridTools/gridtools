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

#include <array>

#include <boost/mpl/vector_c.hpp>

#include "data_store.hpp"
#include "common/data_store_field_metafunctions.hpp"
#include "common/storage_info_interface.hpp"

namespace gridtools {

    /**
     * @brief data_store_field implementation. This struct provides a pack of data_stores. The different coordinates
     * can have arbitrary sizes. So it is more flexible than a 2-dimensional array of data_stores.
     * @tparam DataStore data_store type
     * @tparam N variadic list of coordinate sizes (e.g., 1,2,3 --> first coordinate contains 1 data_store, second 2,
     * third 3)
     */
    template < typename DataStore, unsigned... N >
    struct data_store_field {
        static_assert(is_data_store< DataStore >::value, "Passed type is no data_store type");
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

        /**
         * @brief data_store_field constructor
         */
        constexpr data_store_field() {}

        /**
         * @brief data_store_field constructor
         * @param info storage info that contains size, halo information, etc.
         */
        constexpr data_store_field(storage_info_t const &info) : m_field(fill_array< DataStore, size >(info)) {}

        /**
         * @brief method that is used to extract a data_store out of a data_store_field
         * @tparam Dim requested coordinate
         * @tparam Snapshot requested snapshot
         * @return data_store instance
         */
        template < unsigned Dim, unsigned Snapshot >
        DataStore const &get() const {
            static_assert((get_accumulated_data_field_index(Dim, N...) + Snapshot) < size,
                "Data store field out of bounds access");
            // return
            return m_field[get_accumulated_data_field_index(Dim, N...) + Snapshot];
        }

        /**
         * @brief method that is used to extract a data_store out of a data_store_field
         * @param Dim requested coordinate
         * @param Snapshot requested snapshot
         * @return data_store instance
         */
        DataStore const &get(unsigned Dim, unsigned Snapshot) const {
            assert((get_accumulated_data_field_index(Dim, N...) + Snapshot) < size &&
                   "Data store field out of bounds access");
            // return
            return m_field[get_accumulated_data_field_index(Dim, N...) + Snapshot];
        }

        /**
         * @brief method that is used replace a data_store in a data_store_field
         * @tparam Dim requested coordinate
         * @tparam Snapshot requested snapshot
         * @param store data_store that should be inserted into the field
         */
        template < unsigned Dim, unsigned Snapshot >
        void set(DataStore const &store) {
            static_assert((get_accumulated_data_field_index(Dim, N...) + Snapshot) < size,
                "Data store field out of bounds access");
            // check equality of storage infos
            for (auto elem : get_field()) {
                if (elem.valid()) {
                    assert(store.valid() && "Passed invalid data store.");
                    assert(*elem.get_storage_info_ptr() == *store.get_storage_info_ptr() &&
                           "Passed data store cannot be inserted into data store field because storage infos are not compatible.");
                }
            }
            // set data store
            m_field[get_accumulated_data_field_index(Dim, N...) + Snapshot] = store;
        }

        /**
         * @brief method that is used replace a data_store in a data_store_field
         * @param Dim requested coordinate
         * @param Snapshot requested snapshot
         * @param store data_store that should be inserted into the field
         */
        void set(unsigned Dim, unsigned Snapshot, DataStore const &store) {
            assert((get_accumulated_data_field_index(Dim, N...) + Snapshot) < size &&
                   "Data store field out of bounds access");
            // check equality of storage infos
            for (auto elem : get_field()) {
                if (elem.valid()) {
                    assert(store.valid() && "Passed invalid data store.");
                    assert(*elem.get_storage_info_ptr() == *store.get_storage_info_ptr() &&
                           "Passed data store cannot be inserted into data store field because storage infos do not match.");
                }
            }
            // set data store
            m_field[get_accumulated_data_field_index(Dim, N...) + Snapshot] = store;
        }

        /**
         * @brief explicit allocation of the needed space.
         * @param info storage_info
         */
        void allocate(storage_info_t const &info) {
            for (auto &e : m_field)
                e.allocate(info);
        }

        /**
         * @brief reset the data_store_field
         */
        void reset() {
            for (auto &e : m_field)
                e.reset();
        }

        /**
         * @brief check if all elements of the data_store_field are valid.
         * @return true if all elements are valid, false otherwise
         */
        bool valid() const {
            bool res = true;
            for (auto &e : m_field)
                res &= e.valid();
            return res;
        }

        /**
         * @brief get the content of the data_store_field
         * @return the whole field that contains all data_stores
         */
        std::array< DataStore, size > const &get_field() const { return m_field; }

        /**
         * @brief retrieve the sizes of the data_store_field components
         * @return an array that contains the sizes of the data_store_field components
         */
        constexpr std::array< unsigned, sizeof...(N) > get_dim_sizes() const { return {N...}; }

        /**
         * @brief clone all elements of the field to the device
         */
        void clone_to_device() const {
            for (auto &e : this->m_field)
                e.clone_to_device();
        }

        /**
         * @brief clone all elements of the field from the device
         */
        void clone_from_device() const {
            for (auto &e : this->m_field)
                e.clone_from_device();
        }

        /**
         * @brief synchronize all field elements
         */
        void sync() const {
            for (auto &e : this->m_field)
                e.sync();
        }

        /**
         * @brief reactivate all device read write views to all field elements
         */
        void reactivate_device_write_views() const {
            for (auto &e : this->m_field)
                e.reactivate_device_write_views();
        }

        /**
         * @brief reactivate all host read write views to all field elements
         */
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
     *  @brief Implementation of a swap function. E.g., swap<0,0>::with<0,1>(field_view)
     *  will swap the CPU and GPU pointers of storages 0,0 and 0,1.
     *  This operation invalidates the previously created views.
     **/
    template < unsigned Dim_S, unsigned Snapshot_S >
    struct swap {
        template < unsigned Dim_T, unsigned Snapshot_T, typename T, unsigned... N >
        static void with(data_store_field< T, N... > &data_field) {
            static_assert(Dim_S == Dim_T, "Inter-component swap is not allowed.");
            static_assert(
                is_data_store_field< data_store_field< T, N... > >::value, "Passed type is no data_store_field type.");
            typedef typename std::remove_pointer< decltype(
                std::declval< typename data_store_field< T, N... >::data_store_t >().get_storage_ptr()) >::type::ptrs_t
                ptrs_t;
            auto &src = data_field.template get< Dim_S, Snapshot_S >();
            auto &trg = data_field.template get< Dim_T, Snapshot_T >();
            auto tmp_ptrs = src.get_storage_ptr()->template get_ptrs< ptrs_t >();
            src.get_storage_ptr()->template set_ptrs< ptrs_t >(trg.get_storage_ptr()->template get_ptrs< ptrs_t >());
            trg.get_storage_ptr()->template set_ptrs< ptrs_t >(tmp_ptrs);
        }
    };

    /**
     *  @brief Implementation of a cycle function. E.g., cycle<0>(field_view)
     *  move the last data store of component 0 to the first position
     *  and shifting all others one position to the right.
     *  This operation invalidates the previously created views.
     **/
    template < unsigned Dim >
    struct cycle {
        template < int F, typename T, unsigned... N >
        static void by(data_store_field< T, N... > &data_field) {
            static_assert(
                is_data_store_field< data_store_field< T, N... > >::value, "Passed type is no data_store_field type.");
            typedef typename data_store_field< T, N... >::data_store_t data_store_t;
            typedef typename std::remove_pointer< decltype(
                std::declval< data_store_t >().get_storage_ptr()) >::type::ptrs_t ptrs_t;
            int size = get_value_from_pack(Dim, N...);
            unsigned cnt = 0;
            unsigned src = 0;
            unsigned shift = (((F) % size) < 0) ? ((F) % size) + size : ((F) % size);
            ptrs_t src_ptrs = data_field.get(Dim, 0).get_storage_ptr()->template get_ptrs< ptrs_t >();
            while (cnt < size) {
                // calculate the target position
                unsigned trg = (src + shift) % size;
                // get both data_stores
                auto &trg_ds = data_field.get(Dim, trg);
                auto &src_ds = data_field.get(Dim, src);
                // our tmp pointers are the current source pointers
                auto tmp_ptrs = src_ptrs;
                // update the src pointers with the current target pointers
                src_ptrs = trg_ds.get_storage_ptr()->template get_ptrs< ptrs_t >();
                // replace the target pointers with the tmp_ptrs
                trg_ds.get_storage_ptr()->template set_ptrs< ptrs_t >(tmp_ptrs);
                // increase the count and set the new src position
                cnt++;
                src = trg;
            }
        }
    };

    /**
     *  @brief Implementation of a cycle all function. E.g., cycle_all::by<N>(field_view)
     *  shifts all components by N to the right.
     *  This operation invalidates the previously created views.
     **/
    struct cycle_all {
      private:
        template < int I, int M, typename T, unsigned... N >
        static typename boost::enable_if_c< (I == 0), void >::type by_impl(data_store_field< T, N... > &data_field) {
            cycle< (I) >::template by< (M) >(data_field);
        }

        template < int I, int M, typename T, unsigned... N >
        static typename boost::enable_if_c< (I > 0), void >::type by_impl(data_store_field< T, N... > &data_field) {
            cycle< (I) >::template by< (M) >(data_field);
            by_impl< I - 1, M >(data_field);
        }

      public:
        template < int M, typename T, unsigned... N >
        static void by(data_store_field< T, N... > &data_field) {
            by_impl< (sizeof...(N)-1), M >(data_field);
        }
    };
}
