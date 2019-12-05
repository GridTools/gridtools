/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <type_traits>

namespace gridtools {

    /**
     * @file
     * Some C++17 type_traits drop offs. Please refer to C++17 specifications
     * to know more about them.
     */

    template <bool V>
    using bool_constant = std::integral_constant<bool, V>;

    template <class T>
    using negation = bool_constant<!bool(T::value)>;

    template <class...>
    struct conjunction : std::true_type {};
    template <class T>
    struct conjunction<T> : T {};
    template <class T, class... Ts>
    struct conjunction<T, Ts...> : std::conditional<bool(T::value), conjunction<Ts...>, T>::type {};

    template <class...>
    struct disjunction : std::false_type {};
    template <class T>
    struct disjunction<T> : T {};
    template <class T, class... Ts>
    struct disjunction<T, Ts...> : std::conditional<bool(T::value), T, disjunction<Ts...>>::type {};

    template <typename... Ts>
    using void_t = void;
} // namespace gridtools
