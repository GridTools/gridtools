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

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "data_store.hpp"
#include "storage-facility.hpp"

namespace gridtools {
    namespace storage_sid_impl_ {
        template <class T>
        struct bounds_validator_f {
            T *m_begin;
            T *m_end;

            GT_FUNCTION bool operator()(T *ptr) const { return ptr >= m_begin && ptr < m_end; }
        };
    } // namespace storage_sid_impl_

    template <class Storage, class StorageInfo>
    typename Storage::data_t *sid_get_origin(data_store<Storage, StorageInfo> const &obj) {
        if (obj.device_needs_update())
            obj.sync();
        return advanced::get_raw_pointer_of(make_target_view(obj));
    }

    // TODO(anstaf): we can do better here: return a tuple where strides that are known in complile time are replaced by
    // integral_constants
    template <class Storage, class StorageInfo>
    auto sid_get_strides(data_store<Storage, StorageInfo> const &obj) GT_AUTO_RETURN(obj.strides());

    template <class Storage, class StorageInfo>
    storage_sid_impl_::bounds_validator_f<typename Storage::data_t> sid_get_bounds_validator(
        data_store<Storage, StorageInfo> const &obj) {
        auto origin = sid_get_origin(obj);
        return {origin, origin + obj.total_length()};
    }

    template <class Storage, class StorageInfo>
    int_t sid_get_ptr_diff(data_store<Storage, StorageInfo> const &);

    template <class Storage, class StorageInfo>
    StorageInfo sid_get_strides_kind(data_store<Storage, StorageInfo> const &);
} // namespace gridtools