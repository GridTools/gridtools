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

#include "../../meta.hpp"

namespace gridtools::fn::ast {
    template <class...>
    struct constant;

    template <class V>
    struct constant<V> {};

    template <class...>
    struct in;

    template <class I>
    struct in<I> {};

    template <class...>
    struct lambda;

    template <auto F, class... Args>
    struct lambda<meta::val<F>, Args...> {};

    template <class...>
    struct plus;

    template <class L, class R, class... Ts>
    struct plus<L, R, Ts...> {};

    template <class...>
    struct minus;

    template <class T, class... Ts>
    struct minus<T, Ts...> {};

    template <class...>
    struct multiplies;

    template <class L, class R, class... Ts>
    struct multiplies<L, R, Ts...> {};

    template <class...>
    struct divides;

    template <class L, class R, class... Ts>
    struct divides<L, R, Ts...> {};

    template <class...>
    struct make_tuple {};

    template <class...>
    struct tuple_get;

    template <class I, class Tree>
    struct tuple_get<I, Tree> {};

    template <class...>
    struct deref;

    template <class Tree>
    struct deref<Tree> {};

    template <class...>
    struct shifted;

    template <class Tree, auto... Offsets>
    struct shifted<Tree, meta::val<Offsets...>> {};

    template <class...>
    struct inlined;

    template <auto F, class... Trees>
    struct inlined<meta::val<F>, Trees...> {};

    template <class...>
    struct tmp;

    template <auto F, class... Trees>
    struct tmp<meta::val<F>, Trees...> {};
} // namespace gridtools::fn::ast
