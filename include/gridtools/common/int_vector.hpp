/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 *  @file
 * TODO
 */

#include <type_traits>
#ifndef GT_TARGET_ITERATING
//// DON'T USE #pragma once HERE!!!
#ifndef GT_COMMON_INT_VECTOR_HPP_
#define GT_COMMON_INT_VECTOR_HPP_

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

        template <class T>
        struct value_t_merger<T> {
            using type = T;
        };

        template <class T0, class T1>
        struct value_t_merger<T0, T1> {
            using V0 = meta::second<T0>;
            using V1 = meta::second<T1>;
            using K0 = meta::first<T0>;
            using type = meta::list<K0,
                meta::if_<integral_constant<bool, is_integral_constant<V0>::value && is_integral_constant<V1>::value>,
                    decltype(V0{} + V1{}), // TODO this is specific to `plus`
                    std::common_type_t<V0, V1>>>;
        };

        template <class... T>
        using merger_t = typename value_t_merger<T...>::type;
    } // namespace int_vector_impl_

    namespace int_vector {} // namespace int_vector
} // namespace gridtools

#define GT_FILENAME <gridtools/common/int_vector.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif // GT_COMMON_INT_VECTOR_HPP_
#else  // GT_TARGET_ITERATING

namespace gridtools {
    GT_TARGET_NAMESPACE {}

    namespace int_vector {
        GT_TARGET_NAMESPACE {

            namespace int_vector_detail {
                template <class I, class Enable = void, class Key = meta::first<I>, class Type = meta::second<I>>
                struct add_f {
                    template <class First, class Second>
                    GT_TARGET GT_FORCE_INLINE GT_TARGET_CONSTEXPR Type operator()(
                        First &&first, Second &&second) const {
                        return at_key_with_default<Key, std::integral_constant<Type, 0>>(wstd::forward<First>(first)) +
                               at_key_with_default<Key, std::integral_constant<Type, 0>>(wstd::forward<Second>(second));
                    }
                };
                template <class I>
                struct add_f<I, std::enable_if_t<is_integral_constant<meta::second<I>>::value>> {
                    template <class First, class Second>
                    GT_TARGET GT_FORCE_INLINE GT_TARGET_CONSTEXPR meta::second<I> operator()(
                        First &&, Second &&) const {
                        return {};
                    }
                };

                // TODO or the following?

                // template <class I, class Key = meta::first<I>, class Type = meta::second<I>>
                // struct add_f {
                //     template <class First, class Second>
                //     Type operator()(First &&first, Second &&second) const {
                //         if constexpr (is_integral_constant<Type>::value) {
                //             return {};
                //         } else {
                //             return at_key_with_default<Key, std::integral_constant<Type, 0>>(first) +
                //                    at_key_with_default<Key, std::integral_constant<Type, 0>>(second);
                //         }
                //     }
                // };

                template <class First, class Second>
                GT_TARGET GT_FORCE_INLINE GT_TARGET_CONSTEXPR auto plus(First &&first, Second &&second) {
                    using merged_meta_map_t = meta::mp_make<int_vector_impl_::merger_t,
                        meta::concat<hymap::to_meta_map<First>, hymap::to_meta_map<Second>>>;
                    using keys_t = meta::transform<meta::first, merged_meta_map_t>;
                    using generators = meta::transform<add_f, merged_meta_map_t>;
                    return tuple_util::generate<generators, hymap::from_meta_map<merged_meta_map_t>>(
                        std::forward<First>(first), std::forward<Second>(second));
                }
            } // namespace int_vector_detail

            using int_vector_detail::plus;

            template <class First, class... Rest>
            GT_TARGET GT_FORCE_INLINE GT_TARGET_CONSTEXPR auto plus(First && first, Rest && ...rest) {
                return plus(std::forward<First>(first), plus(std::forward<Rest>(rest)...));
            }
        }
    } // namespace int_vector
} // namespace gridtools

#endif // GT_TARGET_ITERATING
