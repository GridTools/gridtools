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
     * Some c++14/c++17 type_traits drop offs. Please refer to C++14/17 specifications
     * to know more about them.
     */

    template <bool V>
    using bool_constant = std::integral_constant<bool, V>;

    template <class T>
    struct negation : bool_constant<!bool(T::value)> {};

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
    struct void_t_impl {
        using type = void;
    };
    template <typename... Ts>
    using void_t = typename void_t_impl<Ts...>::type;

    template <class T>
    using remove_cv_t = typename std::remove_cv<T>::type;
    template <class T>
    using remove_const_t = typename std::remove_const<T>::type;
    template <class T>
    using remove_volatile_t = typename std::remove_volatile<T>::type;
    template <class T>
    using add_cv_t = typename std::add_cv<T>::type;
    template <class T>
    using add_const_t = typename std::add_const<T>::type;
    template <class T>
    using add_volatile_t = typename std::add_volatile<T>::type;
    template <class T>
    using remove_reference_t = typename std::remove_reference<T>::type;
    template <class T>
    using add_lvalue_reference_t = typename std::add_lvalue_reference<T>::type;
    template <class T>
    using add_rvalue_reference_t = typename std::add_rvalue_reference<T>::type;
    template <class T>
    using remove_pointer_t = typename std::remove_pointer<T>::type;
    template <class T>
    using add_pointer_t = typename std::add_pointer<T>::type;
    template <class T>
    using make_signed_t = typename std::make_signed<T>::type;
    template <class T>
    using make_unsigned_t = typename std::make_unsigned<T>::type;
    template <class T>
    using remove_extent_t = typename std::remove_extent<T>::type;
    template <class T>
    using remove_all_extents_t = typename std::remove_all_extents<T>::type;
    template <class T>
    using decay_t = typename std::decay<T>::type;
    template <bool V, class T = void>
    using enable_if_t = typename std::enable_if<V, T>::type;
    template <bool V, class T, class U>
    using conditional_t = typename std::conditional<V, T, U>::type;
    template <class... Ts>
    using common_type_t = typename std::common_type<Ts...>::type;
    template <class T>
    using underlying_type_t = typename std::underlying_type<T>::type;
    template <class T>
    using result_of_t = typename std::result_of<T>::type;
} // namespace gridtools
