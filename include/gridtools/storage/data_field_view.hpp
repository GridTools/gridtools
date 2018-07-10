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

#include <boost/mpl/and.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits.hpp>

#include "../common/gt_assert.hpp"
#include "common/definitions.hpp"
#include "data_store_field.hpp"
#include "data_view.hpp"

namespace gridtools {

    /**
     * @brief data_field_view implementation. This struct provides means to modify contents of
     * gridtools data_store_field containers on arbitrary locations (host, device, etc.).
     * @tparam DataStoreField data store field type
     * @tparam AccessMode access mode (default is read-write)
     */
    template <typename DataStoreField, access_mode AccessMode = access_mode::ReadWrite>
    struct data_field_view {
        GRIDTOOLS_STATIC_ASSERT(is_data_store_field<DataStoreField>::value,
            GT_INTERNAL_ERROR_MSG("Passed type is no data_store_field type"));
        typedef typename DataStoreField::data_store_t data_store_t;
        typedef typename DataStoreField::data_t data_t;
        typedef typename DataStoreField::state_machine_t state_machine_t;
        typedef typename DataStoreField::storage_info_t storage_info_t;
        static const uint_t num_of_storages = DataStoreField::num_of_storages;
        static const uint_t num_of_components = DataStoreField::num_of_components;
        const static access_mode mode = AccessMode;

        data_t *m_raw_ptrs[num_of_storages];
        state_machine_t *m_state_machines[num_of_storages];
        storage_info_t const *m_storage_infos[num_of_components];
        uint_t m_offsets[num_of_components];
        bool m_device_view;

        /**
         * @brief data_field_view constructor. This constructor is normally not called by the user because it is more
         * convenient to use the provided make functions.
         * @param data_ptrs list of pointers to the data
         * @param info_ptrs list of pointers to the storage_infos
         * @param state_machines list of pointers to the state machines
         * @param offsets list of accumulated offsets (e.g., coordinate sizes 1,2,3 will result in offsets 0, 1, 3)
         * @param device_view true if device view, false otherwise
         */
        data_field_view(data_t *data_ptrs[num_of_storages],
            storage_info_t const *info_ptrs[num_of_components],
            state_machine_t *state_machines[num_of_storages],
            uint_t offsets[num_of_components],
            bool device_view)
            : m_device_view(device_view) {
            for (uint_t i = 0; i < num_of_storages; ++i)
                m_raw_ptrs[i] = data_ptrs[i];
            for (uint_t i = 0; i < num_of_components; ++i)
                m_storage_infos[i] = info_ptrs[i];
            for (uint_t i = 0; i < num_of_storages; ++i)
                m_state_machines[i] = state_machines[i];
            for (uint_t i = 0; i < num_of_components; ++i)
                m_offsets[i] = offsets[i];
        }

        /**
         * @brief get a view to a snapshot in a given coordinate.
         * @tparam Dim requested coordinate
         * @tparam Snapshot requested snapshot
         * @return data_view to the queried data_store
         */
        template <uint_t Dim, uint_t Snapshot>
        GT_FUNCTION data_view<data_store_t, AccessMode> get() const {
            return data_view<data_store_t, AccessMode>(m_raw_ptrs[m_offsets[Dim] + Snapshot],
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
        GT_FUNCTION data_view<data_store_t, AccessMode> get(uint_t Dim, uint_t Snapshot) const {
            return data_view<data_store_t, AccessMode>(m_raw_ptrs[m_offsets[Dim] + Snapshot],
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
            for (uint_t i = 0; i < num_of_components - 1; ++i) {
                for (uint_t j = 0; j < m_offsets[i + 1] - m_offsets[i]; ++j) {
                    res &= get(i, j).valid();
                }
            }
            return res;
        }
    };

    /// @brief simple metafunction to check if a type is a data_field_view
    template <typename T>
    struct is_data_field_view : boost::mpl::false_ {};

    template <typename T, access_mode AccessMode>
    struct is_data_field_view<data_field_view<T, AccessMode>> : boost::mpl::true_ {};

    namespace advanced {
        template <typename T, access_mode AccessMode>
        auto storage_info_raw_ptr(data_field_view<T, AccessMode> const &src) GT_AUTO_RETURN(src.m_storage_infos[0]);
    }
} // namespace gridtools
