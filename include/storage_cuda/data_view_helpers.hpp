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

#include <assert.h>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/utility.hpp>

#include "../common/data_store.hpp"
#include "../common/data_view.hpp"
#include "storage.hpp"
#include "storage_info.hpp"

namespace gridtools {

    // functions used to create views to host data stores (read-write/read-only)
    template < bool ReadOnly = false, typename CudaDataStore >
    typename boost::enable_if< is_cuda_storage< typename CudaDataStore::storage_t >,
        data_view< CudaDataStore, ReadOnly > >::type
    make_host_view(CudaDataStore &ds) {
        assert(ds.valid() && "Cannot create a data_view to an invalid data_store");
        if (!ReadOnly)
            ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu = true;
        return data_view< CudaDataStore, ReadOnly >(ds.get_storage_ptr()->get_cpu_ptr(),
            ds.get_storage_info_ptr(),
            ds.get_storage_ptr()->get_state_machine_ptr(),
            false);
    }

    template < bool ReadOnly = false, typename CudaDataStore >
    typename boost::enable_if< is_cuda_storage< typename CudaDataStore::storage_t >,
        data_view< CudaDataStore, ReadOnly > >::type
    make_device_view(CudaDataStore &ds) {
        assert(ds.valid() && "Cannot create a data_view to an invalid data_store");
        if (!ReadOnly)
            ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu = true;
        return data_view< CudaDataStore, ReadOnly >(ds.get_storage_ptr()->get_gpu_ptr(),
            ds.get_storage_info_ptr()->get_gpu_ptr(),
            ds.get_storage_ptr()->get_state_machine_ptr(),
            true);
    }

    // function that can be used to check if a view is valid
    template < typename DataStore, typename DV >
    typename boost::enable_if<
        boost::mpl::and_< is_cuda_storage< typename DataStore::storage_t >, is_data_store< DataStore > >,
        bool >::type
    valid(DataStore const &d, DV const &v) {
        // if the storage is not valid return false
        if (!d.valid())
            return false;
        // if ptrs do not match anymore return false
        if ((v.m_raw_ptrs[0] != d.get_storage_ptr()->get_gpu_ptr()) &&
            (v.m_raw_ptrs[0] != d.get_storage_ptr()->get_cpu_ptr()))
            return false;
        // check if we have a device view
        const bool device_view = (v.m_raw_ptrs[0] == d.get_storage_ptr()->get_cpu_ptr()) ? false : true;
        // read-only? if yes, take early exit
        if (DV::read_only)
            return device_view ? !d.get_storage_ptr()->get_state_machine_ptr()->m_dnu
                               : !d.get_storage_ptr()->get_state_machine_ptr()->m_hnu;
        // get storage state
        return device_view ? ((d.get_storage_ptr()->get_state_machine_ptr()->m_hnu) && 
                !(d.get_storage_ptr()->get_state_machine_ptr()->m_dnu) && 
                (d.get_storage_ptr()->get_state_machine_ptr()->m_od)) : 
            (!(d.get_storage_ptr()->get_state_machine_ptr()->m_hnu) && 
                (d.get_storage_ptr()->get_state_machine_ptr()->m_dnu) && 
                !(d.get_storage_ptr()->get_state_machine_ptr()->m_od));
    }
}
