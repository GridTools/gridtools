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

#include <boost/preprocessor.hpp>

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

#define GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR(op, type)                                                           \
    template <class T, T V>                                                                                            \
    constexpr GT_FUNCTION integral_constant<decltype(op V), (op V)> operator op(integral_constant<T, V> &&) noexcept { \
        return {};                                                                                                     \
    }                                                                                                                  \
    static_assert(1, "")

    GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR(+, T);
    GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR(-, T);
    GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR(~, T);
    GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR(!, bool);

#undef GT_INTEGRAL_CONSTANT_DEFINE_UNARY_OPERATOR

#define GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(op, type)                            \
    template <class T, T TV, class U, U UV>                                              \
    constexpr GT_FUNCTION integral_constant<decltype(TV op UV), (TV op UV)> operator op( \
        integral_constant<T, TV>, integral_constant<U, UV>) noexcept {                   \
        return {};                                                                       \
    }                                                                                    \
    static_assert(1, "")

    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(+, (std::common_type_t<T, U>));
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(-, (std::common_type_t<T, U>));
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(*, (std::common_type_t<T, U>));
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(/, (std::common_type_t<T, U>));
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(%, (std::common_type_t<T, U>::type));
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(==, bool);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(!=, bool);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(<, bool);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(<=, bool);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(>, bool);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(>=, bool);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(&, (std::common_type_t<T, U>));
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(|, (std::common_type_t<T, U>));
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(<<, (std::common_type_t<T, U>));
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(>>, (std::common_type_t<T, U>));
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(&&, bool);
    GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR(||, bool);

#undef GT_INTEGRAL_CONSTANT_DEFINE_BINARY_OPERATOR

    namespace literals {
        namespace literals_impl_ {

            using literal_int_t = int;

            template <literal_int_t Base>
            constexpr literal_int_t to_int(char c);

            template <>
            constexpr literal_int_t to_int<2>(char c) {
                return c == '0' ? 0 : c == '1' ? 1 : throw "invalid binary _c literal";
            };

            template <>
            constexpr literal_int_t to_int<8>(char c) {
                return c >= '0' && c <= '7' ? c - '0' : throw "invalid octal _c literal";
            };
            template <>
            constexpr literal_int_t to_int<10>(char c) {
                return c >= '0' && c <= '9' ? c - '0' : throw "invalid decimal _c literal";
            };

            template <>
            constexpr literal_int_t to_int<16>(char c) {
                return c >= 'A' && c <= 'F'
                           ? c - 'A' + 10
                           : c >= 'a' && c <= 'f' ? c - 'a' + 10
                                                  : c >= '0' && c <= '9' ? c - '0' : throw "invalid hex _c literal";
            };

            template <literal_int_t Base>
            constexpr literal_int_t parse(char const *first, char const *last) {
                return *last == '\'' ? parse<Base>(first, last - 1)
                                     : to_int<Base>(*last) + (first == last ? 0 : parse<Base>(first, last - 1) * Base);
            }

            template <literal_int_t Base, char... Chars>
            struct digits_parser {
                constexpr static char digits[sizeof...(Chars)] = {Chars...};
                constexpr static literal_int_t value = parse<Base>(digits, digits + sizeof...(Chars) - 1);
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
        } // namespace literals_impl_

        template <char... Chars>
        constexpr GT_FUNCTION integral_constant<literals_impl_::literal_int_t, literals_impl_::parser<Chars...>::value>
        operator"" _c() {
            return {};
        }
    } // namespace literals
} // namespace gridtools

#define GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(v) gridtools::integral_constant<decltype(v), v>()
