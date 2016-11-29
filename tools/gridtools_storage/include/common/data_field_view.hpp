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

#include <boost/mpl/if.hpp>

#include "data_view.hpp"
#include "defs.hpp"

namespace gridtools {

    template < typename DataStoreField, bool ReadOnly >
    struct data_field_view {
        using DataType = typename DataStoreField::data_t;
        using StateMachine = typename DataStoreField::state_machine_t;
        using StorageInfo = typename DataStoreField::storage_info_t;
        static const unsigned N = DataStoreField::size;
        static const unsigned Dims = DataStoreField::dims;

        DataType *m_raw_ptrs[N];
        StateMachine *m_state_machines[N];
        StorageInfo const *m_storage_infos[Dims];
        unsigned m_offsets[Dims];
        bool m_device_view;

        data_field_view(DataType *data_ptrs[N],
            StorageInfo const *info_ptrs[Dims],
            StateMachine *state_machines[N],
            unsigned offsets[Dims],
            bool device_view)
            : m_device_view(device_view) {
            for (unsigned i = 0; i < N; ++i)
                m_raw_ptrs[i] = data_ptrs[i];
            for (unsigned i = 0; i < Dims; ++i)
                m_storage_infos[i] = info_ptrs[i];
            for (unsigned i = 0; i < N; ++i)
                m_state_machines[i] = state_machines[i];
            for (unsigned i = 0; i < Dims; ++i)
                m_offsets[i] = offsets[i];
        }

        template < unsigned Dim, unsigned Snapshot >
        GT_FUNCTION data_view< typename DataStoreField::data_store_t, ReadOnly > get() const {
            return data_view< typename DataStoreField::data_store_t, ReadOnly >(m_raw_ptrs[m_offsets[Dim] + Snapshot],
                m_storage_infos[Dim],
                m_state_machines[m_offsets[Dim] + Snapshot],
                m_device_view);
        }

        GT_FUNCTION data_view< typename DataStoreField::data_store_t, ReadOnly > get(
            unsigned Dim, unsigned Snapshot) const {
            return data_view< typename DataStoreField::data_store_t, ReadOnly >(m_raw_ptrs[m_offsets[Dim] + Snapshot],
                m_storage_infos[Dim],
                m_state_machines[m_offsets[Dim] + Snapshot],
                m_device_view);
        }

        template < unsigned Dim, unsigned Snapshot, typename... Coords >
        typename boost::mpl::if_c< ReadOnly, DataType const &, DataType & >::type GT_FUNCTION get_value(
            Coords... c) const {
            return get< Dim, Snapshot >()(c...);
        }

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

    template < typename T, bool ReadOnly >
    struct is_data_field_view< data_field_view< T, ReadOnly > > : boost::mpl::true_ {};
}
