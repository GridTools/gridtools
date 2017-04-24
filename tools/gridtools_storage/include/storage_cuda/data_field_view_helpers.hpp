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

#include <type_traits>

#include <boost/utility.hpp>
#include <boost/mpl/and.hpp>

#include "storage.hpp"
#include "storage_info.hpp"
#include "../common/data_store_field.hpp"
#include "../common/data_field_view.hpp"

namespace gridtools {

    // functions used to create views to host data field stores (read-write/read-only)
    template < bool ReadOnly = false,
        typename DataStoreField,
        typename DecayedDSF = typename std::decay< DataStoreField >::type >
    typename boost::enable_if<
        boost::mpl::and_< is_cuda_storage< typename DecayedDSF::storage_t >, is_data_store_field< DecayedDSF > >,
        data_field_view< DecayedDSF, ReadOnly > >::type
    make_field_host_view(DataStoreField &ds) {
        typename DecayedDSF::data_t *ptrs[DecayedDSF::size];
        typename DecayedDSF::state_machine_t *state_ptrs[DecayedDSF::size];
        typename DecayedDSF::storage_info_t const *info_ptrs[DecayedDSF::dims];
        unsigned offsets[DecayedDSF::dims] = {
            0,
        };
        for (unsigned i = 1; i < DecayedDSF::dims; ++i) {
            offsets[i] = offsets[i - 1] + ds.get_dim_sizes()[i - 1];
        }
        for (unsigned i = 0; i < DecayedDSF::dims; ++i) {
            info_ptrs[i] = ds.get_field()[offsets[i]].get_storage_info_ptr();
        }
        for (unsigned i = 0; i < DecayedDSF::size; ++i) {
            ptrs[i] = ds.get_field()[i].get_storage_ptr()->get_cpu_ptr();
            state_ptrs[i] = ds.get_field()[i].get_storage_ptr()->get_state_machine_ptr();
            if (!ReadOnly)
                ds.get_field()[i].get_storage_ptr()->get_state_machine_ptr()->m_dnu = true;
        }
        return data_field_view< DecayedDSF, ReadOnly >(ptrs, info_ptrs, state_ptrs, offsets, false);
    }

    template < bool ReadOnly = false,
        typename DataStoreField,
        typename DecayedDSF = typename std::decay< DataStoreField >::type >
    typename boost::enable_if<
        boost::mpl::and_< is_cuda_storage< typename DecayedDSF::storage_t >, is_data_store_field< DecayedDSF > >,
        data_field_view< DecayedDSF, ReadOnly > >::type
    make_field_device_view(DataStoreField &ds) {
        typename DecayedDSF::data_t *ptrs[DecayedDSF::size];
        typename DecayedDSF::state_machine_t *state_ptrs[DecayedDSF::size];
        typename DecayedDSF::storage_info_t const *info_ptrs[DecayedDSF::dims];
        unsigned offsets[DecayedDSF::dims] = {
            0,
        };
        for (unsigned i = 1; i < DecayedDSF::dims; ++i) {
            offsets[i] = offsets[i - 1] + ds.get_dim_sizes()[i - 1];
        }
        for (unsigned i = 0; i < DecayedDSF::dims; ++i) {
            info_ptrs[i] = ds.get_field()[offsets[i]].get_storage_info_ptr()->get_gpu_ptr();
        }
        for (unsigned i = 0; i < DecayedDSF::size; ++i) {
            ptrs[i] = ds.get_field()[i].get_storage_ptr()->get_gpu_ptr();
            state_ptrs[i] = ds.get_field()[i].get_storage_ptr()->get_state_machine_ptr();
            if (!ReadOnly)
                ds.get_field()[i].get_storage_ptr()->get_state_machine_ptr()->m_hnu = true;
        }
        return data_field_view< DecayedDSF, ReadOnly >(ptrs, info_ptrs, state_ptrs, offsets, true);
    }

    // function that can be used to check if a view is valid
    template < typename DataStoreField,
        typename DataFieldView,
        typename DecayedDSF = typename std::decay< DataStoreField >::type,
        typename DecayedDFV = typename std::decay< DataFieldView >::type >
    typename boost::enable_if<
        boost::mpl::and_< is_cuda_storage< typename DecayedDSF::storage_t >, is_data_store_field< DecayedDSF > >,
        bool >::type
    valid(DataStoreField const &ds, DataFieldView const &dv) {
        static_assert(is_data_field_view< DecayedDFV >::value, "Passed type is no data_field_view type");
        bool res = true;
        unsigned i = 0;
        for (unsigned dim = 0; dim < DecayedDSF::dims; ++dim) {
            for (unsigned pos = 0; pos < ds.get_dim_sizes()[dim]; ++pos) {
                res &= valid(ds.get_field()[i], dv.get(dim, pos));
                i++;
            }
        }
        return res;
    }
}
