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

#if __cplusplus >= 201703

#include "list.hpp"
#include "macros.hpp"

namespace gridtools::meta {

    template <auto...>
    struct val {};

    template <auto... Vs>
    constexpr val<Vs...> constant = {};

    template <auto V>
    struct val<V> {
        using type = decltype(V);
        static constexpr type value = V;
    };

    namespace lazy {
        template <class>
        struct vl_split;

        template <template <auto...> class H, auto... Vs>
        struct vl_split<H<Vs...>> {
            using type = list<H<Vs>...>;
        };
    } // namespace lazy
    GT_META_DELEGATE_TO_LAZY(vl_split, class T, T);
} // namespace gridtools::meta

#endif
