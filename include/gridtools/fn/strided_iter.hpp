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
#include "../sid/concept.hpp"

namespace gridtools::fn {
    template <class Key, class Ptr, class Strides>
    struct strided_iter {
        Ptr ptr;
        Strides const &strides;

        constexpr friend strided_iter fn_shift(strided_iter it, auto d, auto val) {
            sid::shift(it.ptr, sid::get_stride_element<Key, decltype(d)>(it.strides), val);
            return it;
        }

        template <class Ptrs>
        strided_iter(Key, Ptrs const &ptrs, Strides const &strides) : ptr(at_key<Key>(ptrs)), strides(strides) {}
    };
    template <class Key, class Ptrs, class Strides>
    strided_iter(Key, Ptrs const &, Strides const &)->strided_iter<Key, element_at<Key, Ptrs>, Strides>;

    template <class Key, class Ptr, class Strides>
    constexpr decltype(auto) fn_deref(strided_iter<Key, Ptr, Strides> const &it) {
        return *it.ptr;
    }
} // namespace gridtools::fn
