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
            GT_FUNCTION GT_CONSTEXPR auto operator()(auto const &...args) const {
                return (host_device::at_key_with_default<Key, integral_constant<int, 0>>(args) + ...);
            }
        };

        template <class First, class Second>
        constexpr GT_FUNCTION auto plus(First &&first, Second &&second) {
            using merged_meta_map_t = meta::mp_make<int_vector_impl_::merger_t,
                meta::concat<hymap::to_meta_map<First>, hymap::to_meta_map<Second>>>;
            using keys_t = meta::transform<meta::first, merged_meta_map_t>;
            using generators = meta::transform<add_f, keys_t>;
            return tuple_util::generate<generators, hymap::from_meta_map<merged_meta_map_t>>(
                std::forward<First>(first), std::forward<Second>(second));
        }

        template <class Vec, class Scalar>
        constexpr GT_FUNCTION auto multiply(Vec &&vec, Scalar scalar) {
            return tuple_util::host_device::transform([scalar](auto v) { return v * scalar; }, std::forward<Vec>(vec));
        }
    } // namespace int_vector_impl_

    namespace int_vector {
        using int_vector_impl_::multiply;
        using int_vector_impl_::plus;
    } // namespace int_vector
} // namespace gridtools
