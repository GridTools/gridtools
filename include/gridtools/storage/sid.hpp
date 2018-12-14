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
#include "../common/gt_assert.hpp"
#include "../common/host_device.hpp"
#include "../common/layout_map.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "common/storage_info_interface.hpp"
#include "data_store.hpp"
#include "storage-facility.hpp"

namespace gridtools {
    namespace storage_sid_impl_ {
        template <int I, int MaxI>
        GT_META_DEFINE_ALIAS(stride_type,
            meta::if_c,
            ((I < 0),
                integral_constant<int_t, 0>,
                GT_META_CALL(meta::if_c, (I == MaxI, integral_constant<int_t, 1>, int_t))));

        template <class I, class Res>
        struct stride_generator_f {
            template <class Src>
            Res operator()(Src const &) {}
        };

        template <class I, int V>
        struct stride_generator_f<I, integral_constant<int_t, V>> {
            template <class Src>
            integral_constant<int_t, V> operator()(Src const &src) {
                assert(src[I::value] == V);
                return {};
            }
        };

        template <class I>
        struct stride_generator_f<I, int_t> {
            template <class Src>
            int_t operator()(Src const &src) {
                assert(src[I::value] != 0);
                assert(src[I::value] != 1);
                return (int)src[I::value];
            }
        };
    } // namespace storage_sid_impl_

    template <class Storage, class StorageInfo>
    typename Storage::data_t *sid_get_origin(data_store<Storage, StorageInfo> const &obj) {
        if (obj.device_needs_update())
            obj.sync();
        return advanced::get_raw_pointer_of(make_target_view(obj));
    }

    template <class Storage,
        uint_t Id,
        int... Is,
        class Halo,
        class Alignment,
        int MaxI = int(layout_map<Is...>::unmasked_length - 1),
        class Res = tuple<GT_META_CALL(storage_sid_impl_::stride_type, (Is, MaxI))...>>
    Res sid_get_strides(
        data_store<Storage, storage_info_interface<Id, layout_map<Is...>, Halo, Alignment>> const &obj) {
        using indices_t = GT_META_CALL(meta::make_indices_c, sizeof...(Is));
        using generators_t = GT_META_CALL(meta::transform, (storage_sid_impl_::stride_generator_f, indices_t, Res));
        return tuple_util::generate<generators_t, Res>(obj.strides());
    }

    template <class Storage, class StorageInfo>
    StorageInfo sid_get_strides_kind(data_store<Storage, StorageInfo> const &);

    template <class Storage, class StorageInfo>
    int_t sid_get_ptr_diff(data_store<Storage, StorageInfo> const &);
} // namespace gridtools
