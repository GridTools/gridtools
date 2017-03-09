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

#include <boost/utility.hpp>

#include "../common/data_store.hpp"
#include "../common/data_view.hpp"
#include "storage.hpp"
#include "storage_info.hpp"

namespace gridtools {

    // functions used to create views to host data stores (read-write/read-only)
    template < bool ReadOnly = false, typename DataStore >
    typename boost::enable_if<
        boost::mpl::and_< is_host_storage< typename DataStore::storage_t >, is_data_store< DataStore > >,
        data_view< DataStore, ReadOnly > >::type
    make_host_view(DataStore &ds) {
        assert(ds.valid() && "Cannot create a data_view to an invalid data_store");
        return data_view< DataStore, ReadOnly >(ds.get_storage_ptr()->get_cpu_ptr(),
            ds.get_storage_info_ptr(),
            ds.get_storage_ptr()->get_state_machine_ptr(),
            false);
    }

    // function that can be used to check if a view is valid
    template < typename DataStore, typename DataView >
    typename boost::enable_if<
        boost::mpl::and_< is_host_storage< typename DataStore::storage_t >, is_data_store< DataStore > >,
        bool >::type
    valid(DataStore const &ds, DataView const &dv) {
        static_assert(is_data_view<DataView>::value, "Passed type is no data_view type");
        return ds.valid() && (dv.m_raw_ptrs[0] == ds.get_storage_ptr()->get_cpu_ptr()) &&
               (dv.m_storage_info && ds.get_storage_info_ptr());
    }
}
