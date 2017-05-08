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

#include <boost/mpl/if.hpp>
#include <boost/mpl/and.hpp>
#include <boost/type_traits.hpp>

#include "data_store_field.hpp"
#include "data_view.hpp"
#include "common/definitions.hpp"

namespace gridtools {

    /**
     * @brief data_field_view implementation. This struct provides means to modify contents of
     * gridtools data_store_field containers on arbitrary locations (host, device, etc.).
     * @tparam DataStoreField data store field type
     * @tparam AccessMode access mode (default is read-write)
     */
    template < typename DataStoreField, access_mode AccessMode = access_mode::ReadWrite >
    struct data_field_view {
        static_assert(is_data_store_field< DataStoreField >::value, "Passed type is no data_store_field type");
        typedef typename DataStoreField::data_store_t data_store_t;
        typedef typename DataStoreField::data_t data_t;
        typedef typename DataStoreField::state_machine_t state_machine_t;
        typedef typename DataStoreField::storage_info_t storage_info_t;
        static const unsigned view_size = DataStoreField::size;
        static const unsigned Dims = DataStoreField::dims;
        const static access_mode mode = AccessMode;

        data_t *m_raw_ptrs[view_size];
        state_machine_t *m_state_machines[view_size];
        storage_info_t const *m_storage_infos[Dims];
        unsigned m_offsets[Dims];
        bool m_device_view;

        /**
         * @brief data_field_view constructor
         */
        data_field_view() {}

        /**
         * @brief data_field_view constructor. This constructor is normally not called by the user because it is more
         * convenient to use the provided make functions.
         * @param data_ptrs list of pointers to the data
         * @param info_ptr list of pointers to the storage_infos
         * @param state_ptr list of pointers to the state machines
         * @param offsets list of accumulated offsets (e.g., coordinate sizes 1,2,3 will result in offsets 0, 1, 3)
         * @param device_view true if device view, false otherwise
         */
        data_field_view(data_t *data_ptrs[view_size],
            storage_info_t const *info_ptrs[Dims],
            state_machine_t *state_machines[view_size],
            unsigned offsets[Dims],
            bool device_view)
            : m_device_view(device_view) {
            for (unsigned i = 0; i < view_size; ++i)
                m_raw_ptrs[i] = data_ptrs[i];
            for (unsigned i = 0; i < Dims; ++i)
                m_storage_infos[i] = info_ptrs[i];
            for (unsigned i = 0; i < view_size; ++i)
                m_state_machines[i] = state_machines[i];
            for (unsigned i = 0; i < Dims; ++i)
                m_offsets[i] = offsets[i];
        }

        /**
         * @brief get a view to a snapshot in a given coordinate.
         * @tparam Dim requested coordinate
         * @tparam Snapshot requested snapshot
         * @return data_view to the queried data_store
         */
        template < unsigned Dim, unsigned Snapshot >
        GT_FUNCTION data_view< data_store_t, AccessMode > get() const {
            return data_view< data_store_t, AccessMode >(m_raw_ptrs[m_offsets[Dim] + Snapshot],
                m_storage_infos[Dim],
                m_state_machines[m_offsets[Dim] + Snapshot],
                m_device_view);
        }

        /**
         * @brief get a view to a snapshot in a given coordinate.
         * @param Dim requested coordinate
         * @param Snapshot requested snapshot
         * @return data_view to the queried data_store
         */
        GT_FUNCTION data_view< data_store_t, AccessMode > get(unsigned Dim, unsigned Snapshot) const {
            return data_view< data_store_t, AccessMode >(m_raw_ptrs[m_offsets[Dim] + Snapshot],
                m_storage_infos[Dim],
                m_state_machines[m_offsets[Dim] + Snapshot],
                m_device_view);
        }

        /**
         * @brief Check if view contains valid pointers, and simple state machine checks.
         * Be aware that this is not a full check. In order to check if a view is in a
         * consistent state use check_consistency function.
         * @return true if pointers and state is correct, otherwise false
         */
        bool valid() const {
            bool res = true;
            for (unsigned i = 0; i < Dims - 1; ++i) {
                for (unsigned j = 0; j < m_offsets[i + 1] - m_offsets[i]; ++j) {
                    res &= get(i, j).valid();
                }
            }
            return res;
        }
    };

    template < typename T >
    struct is_data_field_view : boost::mpl::false_ {};

    template < typename T, access_mode AccessMode >
    struct is_data_field_view< data_field_view< T, AccessMode > > : boost::mpl::true_ {};
}
