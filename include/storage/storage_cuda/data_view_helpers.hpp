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
#include <boost/utility.hpp>

#include "../data_store.hpp"
#include "../data_view.hpp"
#include "storage.hpp"
#include "storage_info.hpp"

namespace gridtools {

    /**
     * @brief function used to create host views to data stores (read-write/read-only).
     * @tparam AccessMode access mode information (default is read-write).
     * @param ds data store
     * @return a host view to the given data store.
     */
    template < access_mode AccessMode = access_mode::ReadWrite,
        typename CudaDataStore,
        typename DecayedCDS = typename boost::decay< CudaDataStore >::type >
    typename boost::enable_if< boost::mpl::and_< is_cuda_storage< typename DecayedCDS::storage_t >,
                                   is_cuda_storage_info< typename DecayedCDS::storage_info_t >,
                                   is_data_store< DecayedCDS > >,
        data_view< DecayedCDS, AccessMode > >::type
    make_host_view(CudaDataStore const &ds) {
        assert(ds.valid() && "Cannot create a data_view to an invalid data_store");
        if (AccessMode != access_mode::ReadOnly) {
            assert(!ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu && "There is already an active read-write "
                                                                            "device view. Synchronization is needed "
                                                                            "before constructing the view.");
            ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu = true;
        }
        return data_view< DecayedCDS, AccessMode >(ds.get_storage_ptr()->get_cpu_ptr(),
            ds.get_storage_info_ptr(),
            ds.get_storage_ptr()->get_state_machine_ptr(),
            false);
    }

    /**
     * @brief function used to create device views to data stores (read-write/read-only).
     * @tparam AccessMode access mode information (default is read-write).
     * @param ds data store
     * @return a device view to the given data store.
     */
    template < access_mode AccessMode = access_mode::ReadWrite,
        typename CudaDataStore,
        typename DecayedCDS = typename boost::decay< CudaDataStore >::type >
    typename boost::enable_if< boost::mpl::and_< is_cuda_storage< typename DecayedCDS::storage_t >,
                                   is_cuda_storage_info< typename DecayedCDS::storage_info_t >,
                                   is_data_store< DecayedCDS > >,
        data_view< DecayedCDS, AccessMode > >::type
    make_device_view(CudaDataStore const &ds) {
        assert(ds.valid() && "Cannot create a data_view to an invalid data_store");
        if (AccessMode != access_mode::ReadOnly) {
            assert(!ds.get_storage_ptr()->get_state_machine_ptr()->m_dnu && "There is already an active read-write "
                                                                            "host view. Synchronization is needed "
                                                                            "before constructing the view.");
            ds.get_storage_ptr()->get_state_machine_ptr()->m_hnu = true;
        }
        return data_view< DecayedCDS, AccessMode >(ds.get_storage_ptr()->get_gpu_ptr(),
            ds.get_storage_info_ptr()->get_gpu_ptr(),
            ds.get_storage_ptr()->get_state_machine_ptr(),
            true);
    }

    /**
     * @brief function that can be used to check if a view is in a consistent state
     * @param ds data store
     * @param dv data view
     * @return true if the given view is in a valid state and can be used safely.
     */
    template < typename DataStore,
        typename DataView,
        typename DecayedDS = typename boost::decay< DataStore >::type,
        typename DecayedDV = typename boost::decay< DataView >::type >
    typename boost::enable_if< boost::mpl::and_< is_cuda_storage< typename DecayedDS::storage_t >,
                                   is_cuda_storage_info< typename DecayedDS::storage_info_t >,
                                   is_data_store< DecayedDS > >,
        bool >::type
    check_consistency(DataStore const &d, DataView const &v) {
        static_assert(is_data_view< DecayedDV >::value, "Passed type is no data_view type");
        // if the storage is not valid return false
        if (!d.valid())
            return false;
        // if ptrs do not match anymore return false
        if ((advanced::get_raw_pointer_of(v) != d.get_storage_ptr()->get_gpu_ptr()) &&
            (advanced::get_raw_pointer_of(v) != d.get_storage_ptr()->get_cpu_ptr()))
            return false;
        // check if we have a device view
        const bool device_view = (advanced::get_raw_pointer_of(v) == d.get_storage_ptr()->get_cpu_ptr()) ? false : true;
        // read-only? if yes, take early exit
        if (DecayedDV::mode == access_mode::ReadOnly)
            return device_view ? !d.get_storage_ptr()->get_state_machine_ptr()->m_dnu
                               : !d.get_storage_ptr()->get_state_machine_ptr()->m_hnu;
        // get storage state
        return device_view ? ((d.get_storage_ptr()->get_state_machine_ptr()->m_hnu) &&
                                 !(d.get_storage_ptr()->get_state_machine_ptr()->m_dnu))
                           : (!(d.get_storage_ptr()->get_state_machine_ptr()->m_hnu) &&
                                 (d.get_storage_ptr()->get_state_machine_ptr()->m_dnu));
    }
}
