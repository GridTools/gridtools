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

/**
 *  @file
 * TODO
 */

#include <type_traits>

#include "../meta.hpp"
#include "defs.hpp"
#include "gridtools/meta/debug.hpp"
#include "host_device.hpp"
#include "hymap.hpp"
#include "integral_constant.hpp"
#include "tuple.hpp"
#include "tuple_util.hpp"

namespace gridtools {

    namespace int_vector_impl_ {
        template <class... Ts>
        struct value_t_merger;

        template <template <class...> class Item, class Key, class... Vs>
        struct value_t_merger<Item<Key, Vs>...> {
            using type = Item<Key, std::decay_t<decltype((Vs{} + ...))>>;
        };

        template <class... T>
        using merger_t = typename value_t_merger<T...>::type;

        template <class Key>
        struct add_f {
            template <class... Ts>
            GT_FUNCTION GT_CONSTEXPR decltype(auto) operator()(Ts const &...args) const {
                return (host_device::at_key_with_default<Key, integral_constant<int, 0>>(args) + ...);
            }
        };

        template <class First, class Second>
        GT_FUNCTION GT_CONSTEXPR auto plus(First &&first, Second &&second) {
            using merged_meta_map_t = meta::mp_make<int_vector_impl_::merger_t,
                meta::concat<hymap::to_meta_map<First>, hymap::to_meta_map<Second>>>;
            using keys_t = meta::transform<meta::first, merged_meta_map_t>;
            using generators = meta::transform<add_f, keys_t>;
            return tuple_util::host_device::generate<generators, hymap::from_meta_map<merged_meta_map_t>>(
                std::forward<First>(first), std::forward<Second>(second));
        }

        template <class Vec, class Scalar>
        GT_FUNCTION GT_CONSTEXPR auto multiply(Vec &&vec, Scalar scalar) {
            return tuple_util::host_device::transform([scalar](auto v) { return v * scalar; }, std::forward<Vec>(vec));
        }

        template <class I, class Enable = void>
        struct is_integral_zero : std::false_type {};

        template <class I>
        struct is_integral_zero<I,
            std::enable_if_t<is_integral_constant<meta::second<I>>::value && meta::second<I>::value == 0>>
            : std::true_type {};

        template <class Key>
        struct at_key_f {
            template <class T>
            GT_FUNCTION GT_CONSTEXPR decltype(auto) operator()(T const &arg) const {
                return host_device::at_key<Key>(arg);
            }
        };

        template <class Vec>
        GT_FUNCTION GT_CONSTEXPR auto normalize(Vec &&vec) {
            using filtered_map_t = meta::filter<meta::not_<is_integral_zero>::apply, hymap::to_meta_map<Vec>>;
            using keys_t = meta::transform<meta::first, filtered_map_t>;
            using generators = meta::transform<at_key_f, keys_t>;
            return tuple_util::host_device::generate<generators, hymap::from_meta_map<filtered_map_t>>(
                std::forward<Vec>(vec));
        }
    } // namespace int_vector_impl_

    template <class T>
    struct is_int_vector : std::true_type {
    }; // TODO proper type trait: is_hymap and (is_integral or is_gr_integral_constant)

    template <class T>
    inline constexpr bool is_int_vector_v = is_int_vector<T>::value;

    namespace int_vector {
        using int_vector_impl_::multiply;
        using int_vector_impl_::normalize;
        using int_vector_impl_::plus;

        namespace arithmetic {
            template <class Vec,
                class Scalar,
                std::enable_if_t<is_int_vector_v<Vec> &&
                                     (std::is_integral_v<Scalar> || is_integral_constant<Scalar>::value),
                    bool> = true>
            GT_FUNCTION GT_CONSTEXPR auto operator*(Vec &&vec, Scalar scalar) {
                return multiply(std::forward<Vec>(vec), scalar);
            }

            template <class First,
                class Second,
                std::enable_if_t<is_int_vector_v<First> && is_int_vector_v<Second>, bool> = true>
            GT_FUNCTION GT_CONSTEXPR auto operator+(First &&first, Second &&second) {
                return plus(std::forward<First>(first), std::forward<Second>(second));
            }

            template <class Vec, std::enable_if_t<is_int_vector_v<Vec>, bool> = true>
            GT_FUNCTION GT_CONSTEXPR auto operator-(Vec &&vec) {
                return multiply(std::forward<Vec>(vec), integral_constant<int, -1>{}); // TODO not compiling with nvcc
            }

            template <class First,
                class Second,
                std::enable_if_t<is_int_vector_v<First> && is_int_vector_v<Second>, bool> = true>
            GT_FUNCTION GT_CONSTEXPR auto operator-(First &&first, Second &&second) {
                return plus(std::forward<First>(first), -std::forward<Second>(second));
            }

        } // namespace arithmetic
    } // namespace int_vector
} // namespace gridtools
