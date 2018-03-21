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

/**
 * Some c++14/c++17 type_traits drop offs
 */

#pragma once

#include <type_traits>

namespace gridtools {

    template < bool V >
    using bool_constant = std::integral_constant< bool, V >;

    template < class T >
    struct negation : bool_constant< !bool(T::value) > {};

    template < class... >
    struct conjunction : std::true_type {};
    template < class T >
    struct conjunction< T > : T {};
    template < class T, class... Ts >
    struct conjunction< T, Ts... > : std::conditional< bool(T::value), conjunction< Ts... >, T >::type {};

    template < class... >
    struct disjunction : std::false_type {};
    template < class T >
    struct disjunction< T > : T {};
    template < class T, class... Ts >
    struct disjunction< T, Ts... > : std::conditional< bool(T::value), T, disjunction< Ts... > >::type {};

    template < typename... Ts >
    struct void_t_impl {
        using type = void;
    };
    template < typename... Ts >
    using void_t = typename void_t_impl< Ts... >::type;

    template < class T >
    using remove_cv_t = typename std::remove_cv< T >::type;
    template < class T >
    using remove_const_t = typename std::remove_const< T >::type;
    template < class T >
    using remove_volatile_t = typename std::remove_volatile< T >::type;
    template < class T >
    using add_cv_t = typename std::add_cv< T >::type;
    template < class T >
    using add_const_t = typename std::add_const< T >::type;
    template < class T >
    using add_volatile_t = typename std::add_volatile< T >::type;
    template < class T >
    using remove_reference_t = typename std::remove_reference< T >::type;
    template < class T >
    using add_lvalue_reference_t = typename std::add_lvalue_reference< T >::type;
    template < class T >
    using add_rvalue_reference_t = typename std::add_rvalue_reference< T >::type;
    template < class T >
    using remove_pointer_t = typename std::remove_pointer< T >::type;
    template < class T >
    using add_pointer_t = typename std::add_pointer< T >::type;
    template < class T >
    using make_signed_t = typename std::make_signed< T >::type;
    template < class T >
    using make_unsigned_t = typename std::make_unsigned< T >::type;
    template < class T >
    using remove_extent_t = typename std::remove_extent< T >::type;
    template < class T >
    using remove_all_extents_t = typename std::remove_all_extents< T >::type;
    template < std::size_t Len, class... Types >
    using aligned_union_t = typename std::aligned_union< Len, Types... >::type;
    template < class T >
    using decay_t = typename std::decay< T >::type;
    template < bool V, class T = void >
    using enable_if_t = typename std::enable_if< V, T >::type;
    template < bool V, class T, class U >
    using conditional_t = typename std::conditional< V, T, U >::type;
    template < class... Ts >
    using common_type_t = typename std::common_type< Ts... >::type;
    template < class T >
    using underlying_type_t = typename std::underlying_type< T >::type;
    template < class T >
    using result_of_t = typename std::result_of< T >::type;
}
