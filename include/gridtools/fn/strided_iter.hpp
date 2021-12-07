/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include "../common/hymap.hpp"
#include "../meta.hpp"
#include "../sid/concept.hpp"
#include "../sid/multi_shift.hpp"
#include "builtins.hpp"

namespace gridtools::fn {
    namespace strided_iter_impl_ {
        template <class... Ts>
        using is_pair = std::bool_constant<sizeof...(Ts) == 2>;

        template <class...>
        struct merge_offsets;

        template <template <class...> class L, template <auto...> class H, auto Dim, auto... Vals>
        struct merge_offsets<L<H<Dim>, H<Vals>>...> {
            using type = L<std::decay_t<decltype(Dim)>, integral_constant<int, (... + Vals)>>;
        };

        template <class T>
        using not_zero = std::bool_constant<meta::second<T>::value != 0>;

        template <class... Offsets>
        using offset_map = hymap::from_meta_map<meta::filter<not_zero,
            meta::mp_make<meta::force<merge_offsets>::apply,
                meta::group<is_pair, meta::list, meta::list<Offsets...>>>>>;
    } // namespace strided_iter_impl_

    template <class Key, class Ptr, class Strides>
    struct strided_iter {
        Ptr ptr;
        Strides const &strides;

        template <class Dim, class Offset>
        auto sid_access(Dim, Offset offset) const {
            return *sid::shifted(ptr, sid::get_stride_element<Key, Dim>(strides), offset);
        }

        template <auto... Offsets>
        friend std::enable_if_t<sizeof...(Offsets) % 2 == 0, strided_iter> fn_builtin(
            builtins::shift<Offsets...>, strided_iter it) {
            sid::multi_shift<Key>(it.ptr, it.strides, strided_iter_impl_::offset_map<meta::val<Offsets>...>());
            return it;
        }

        template <class Ptrs>
        strided_iter(Key, Ptrs const &ptrs, Strides const &strides) : ptr(at_key<Key>(ptrs)), strides(strides) {}
    };
    template <class Key, class Ptrs, class Strides>
    strided_iter(Key, Ptrs const &, Strides const &)->strided_iter<Key, element_at<Key, Ptrs>, Strides>;

    template <class Key, class Ptr, class Strides>
    constexpr decltype(auto) fn_builtin(builtins::deref, strided_iter<Key, Ptr, Strides> const &it) {
        return *it.ptr;
    }
} // namespace gridtools::fn
