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

#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits.hpp>

#include "definitions.hpp"
#include "storage_info_interface.hpp"

namespace gridtools {

    // data view implementation for data stores
    template < typename DataStore, bool ReadOnly = false >
    struct data_view {
        static_assert(is_data_store< DataStore >::value, "Passed type is no data_store type");
        typedef typename DataStore::data_t data_t;
        typedef typename DataStore::state_machine_t state_machine_t;
        typedef typename DataStore::storage_info_t storage_info_t;
        const static bool read_only = ReadOnly;
        const static unsigned N = 1;

        data_t *m_raw_ptrs[1];
        state_machine_t *m_state_machine_ptr;
        storage_info_t const *m_storage_info;
        bool m_device_view;

        GT_FUNCTION data_view() {}

        GT_FUNCTION data_view(
            data_t *data_ptr, storage_info_t const *info_ptr, state_machine_t *state_ptr, bool device_view)
            : m_raw_ptrs{data_ptr}, m_state_machine_ptr(state_ptr), m_storage_info(info_ptr),
              m_device_view(device_view) {
            assert(data_ptr && "Cannot create data_view with invalid data pointer");
            assert(info_ptr && "Cannot create data_view with invalid storage info pointer");
        }

        template < typename... Coords >
        typename boost::mpl::if_c< ReadOnly, data_t const &, data_t & >::type GT_FUNCTION operator()(
            Coords... c) const {
            return m_raw_ptrs[0][m_storage_info->index(c...)];
        }

        bool valid() const {
            // ptrs invalid -> view invalid
            if (!m_raw_ptrs[0] || !m_storage_info)
                return false;
            // when used in combination with a host storage the view is always valid as long as the ptrs are
            if (!m_state_machine_ptr)
                return true;
            // read only -> simple check
            if (ReadOnly)
                return m_device_view ? !m_state_machine_ptr->m_dnu : !m_state_machine_ptr->m_hnu;
            // check state machine ptrs
            return m_device_view
                       ? ((m_state_machine_ptr->m_hnu) && !(m_state_machine_ptr->m_dnu) && (m_state_machine_ptr->m_od))
                       : (!(m_state_machine_ptr->m_hnu) && (m_state_machine_ptr->m_dnu) &&
                             !(m_state_machine_ptr->m_od));
        }
    };

    template < typename T >
    struct is_data_view : boost::mpl::false_ {};

    template < typename Storage, bool ReadOnly >
    struct is_data_view< data_view< Storage, ReadOnly > > : boost::mpl::true_ {};
}
