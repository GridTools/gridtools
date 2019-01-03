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

#include <type_traits>

#include "host_device.hpp"

namespace gridtools {
    template <class T, T V>
    struct integral_constant : std::integral_constant<T, V> {
        using type = integral_constant;

#if defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1900
        constexpr GT_FORCE_INLINE integral_constant() noexcept {}
#endif

        constexpr GT_FUNCTION operator T() const noexcept { return V; }
    };

#define GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR(op)                                                                 \
    template <class T, T V>                                                                                            \
    constexpr GT_FUNCTION integral_constant<decltype(op V), (op V)> operator op(integral_constant<T, V> &&) noexcept { \
        return {};                                                                                                     \
    }                                                                                                                  \
    static_assert(1, "")

    GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR(+);
    GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR(-);
    GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR(~);
    GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR(!);

#undef GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR

#define GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(op)                                  \
    template <class T, T TV, class U, U UV>                                              \
    constexpr GT_FUNCTION integral_constant<decltype(TV op UV), (TV op UV)> operator op( \
        integral_constant<T, TV>, integral_constant<U, UV>) noexcept {                   \
        return {};                                                                       \
    }                                                                                    \
    static_assert(1, "")

    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(+);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(-);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(*);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(/);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(%);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(==);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(!=);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(<);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(<=);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(>);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(>=);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(&);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(|);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(<<);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(>>);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(&&);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(||);

#undef GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR

    namespace literals {
        namespace impl_ {

            using literal_int_t = int;

            constexpr literal_int_t to_int(char c) {
                return c >= 'A' && c <= 'F' ? c - 'A' + 10 : c >= 'a' && c <= 'a' ? c - 'a' + 10 : c - '0';
            };

            constexpr literal_int_t parse(literal_int_t base, char const *first, char const *last) {
                return to_int(*last) + (first == last ? 0 : parse(base, first, last - 1) * base);
            }

            template <literal_int_t Base, char... Chars>
            struct digits_parser {
                constexpr static char digits[] = {Chars...};
                constexpr static literal_int_t value = parse(Base, digits, digits + sizeof...(Chars) - 1);
            };

            template <char... Chars>
            struct parser : digits_parser<10, Chars...> {};

            template <>
            struct parser<'0'> : integral_constant<literal_int_t, 0> {};

            template <char... Chars>
            struct parser<'0', Chars...> : digits_parser<8, Chars...> {};

            template <char... Chars>
            struct parser<'0', 'x', Chars...> : digits_parser<16, Chars...> {};

            template <char... Chars>
            struct parser<'0', 'b', Chars...> : digits_parser<2, Chars...> {};
        } // namespace impl_

        template <char... Chars>
        constexpr integral_constant<impl_::literal_int_t, impl_::parser<Chars...>::value> operator"" _c() {
            return {};
        }
    } // namespace literals
} // namespace gridtools
