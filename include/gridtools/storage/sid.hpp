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

#include <cassert>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/utility.hpp"
#include "../common/host_device.hpp"
#include "../common/integral_constant.hpp"
#include "../common/layout_map.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../meta/if.hpp"
#include "../meta/macros.hpp"
#include "../meta/make_indices.hpp"
#include "../meta/transform.hpp"
#include "data_store.hpp"
#include "storage-facility.hpp"

namespace gridtools {
    namespace storage_sid_impl_ {

        enum class dimension_kind { masked, innermost, dynamic };

        constexpr dimension_kind get_dimension_kind(int i, size_t n_dim) {
            return i < 0 ? dimension_kind::masked
                         : i + 1 == n_dim ? dimension_kind::innermost : dimension_kind::dynamic;
        }

        template <dimension_kind Kind>
        struct stride_type;

        template <>
        struct stride_type<dimension_kind::masked> {
            using type = integral_constant<int_t, 0>;
        };

        template <>
        struct stride_type<dimension_kind::innermost> {
            using type = integral_constant<int_t, 1>;
        };

        template <>
        struct stride_type<dimension_kind::dynamic> {
            using type = int_t;
        };

        template <class I, class Res>
        struct stride_generator_f;

        template <class I, int V>
        struct stride_generator_f<I, integral_constant<int_t, V>> {
            using type = stride_generator_f;
            template <class Src>
            integral_constant<int_t, V> operator()(Src const &src) {
                assert(src[I::value] == V);
                return {};
            }
        };

        template <class I>
        struct stride_generator_f<I, int_t> {
            using type = stride_generator_f;
            template <class Src>
            int_t operator()(Src const &src) {
                assert(src[I::value] != 0);
                assert(src[I::value] != 1);
                return (int_t)src[I::value];
            }
        };

        template <class Layout>
        struct convert_strides_f;

        template <int... Is>
        struct convert_strides_f<layout_map<Is...>> {
            using res_t =
                tuple<typename stride_type<get_dimension_kind(Is, layout_map<Is...>::unmasked_length)>::type...>;
            using generators_t = GT_META_CALL(
                meta::transform, (stride_generator_f, GT_META_CALL(meta::make_indices_c, sizeof...(Is)), res_t));

            template <class Src>
            res_t operator()(Src const &src) const {
                return tuple_util::generate<generators_t, res_t>(src);
            }
        };

        struct empty_ptr_diff {
            template <class T>
            friend constexpr GT_FUNCTION T &&operator+(T &&lhs, empty_ptr_diff) {
                return const_expr::forward<T>(lhs);
            }
        };

        template <class Storage, class StorageInfo>
        class host_adapter {
            using impl_t = data_store<Storage, StorageInfo>;
            impl_t m_impl;

            static impl_t const &impl();

            friend typename Storage::data_t *sid_get_origin(host_adapter const &obj) {
                impl_t const &impl = obj.m_impl;
                if (impl.host_needs_update())
                    impl.sync();
                return advanced_get_raw_pointer_of(make_host_view(impl));
            }
            friend decltype(sid_get_strides(impl())) sid_get_strides(host_adapter const &obj) {
                return sid_get_strides(obj.m_impl);
            }
            friend decltype(sid_get_strides_kind(impl())) sid_get_strides_kind(host_adapter const &) { return {}; }
            friend decltype(sid_get_ptr_diff(impl())) sid_get_ptr_diff(host_adapter const &) { return {}; }

          public:
            host_adapter(data_store<Storage, StorageInfo> obj) : m_impl(std::move(obj)) {}
        };
    } // namespace storage_sid_impl_

    /**
     *   The functions below make `data_store` model the `SID` concept
     */
    template <class Storage, class StorageInfo>
    typename Storage::data_t *sid_get_origin(data_store<Storage, StorageInfo> const &obj) {
        if (obj.device_needs_update())
            obj.sync();
        return advanced_get_raw_pointer_of(make_target_view(obj));
    }

    template <class Storage, class StorageInfo>
    auto sid_get_strides(data_store<Storage, StorageInfo> const &obj)
        GT_AUTO_RETURN(storage_sid_impl_::convert_strides_f<typename StorageInfo::layout_t>{}(obj.strides()));

    template <class Storage, class StorageInfo>
    StorageInfo sid_get_strides_kind(data_store<Storage, StorageInfo> const &);

    template <class Storage, class StorageInfo>
    GT_META_CALL(meta::if_c, (StorageInfo::layout_t::unmasked_length == 0, storage_sid_impl_::empty_ptr_diff, int_t))
    sid_get_ptr_diff(data_store<Storage, StorageInfo> const &);

    /**
     *  Returns an object that models the `SID` concept the same way as original object does except that the pointers
     *  are taken from the host view.
     */
    template <class Storage, class StorageInfo>
    storage_sid_impl_::host_adapter<Storage, StorageInfo> as_host(data_store<Storage, StorageInfo> obj) {
        return {std::move(obj)};
    }
} // namespace gridtools
