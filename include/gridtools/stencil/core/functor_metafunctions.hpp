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

#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"
#include "../../meta.hpp"
#include "interval.hpp"
#include "level.hpp"

namespace gridtools {
    namespace stencil {
        namespace core {
            namespace functor_metafunctions_impl_ {
                struct probe {};

                template <class T, class = void>
                struct has_apply : std::false_type {};

                template <class T>
                struct has_apply<T, void_t<decltype(T::apply(std::declval<probe const &>()))>> : std::true_type {};

                template <class From>
                struct to_resolver {
                    template <class R, class To>
                    static To select(R (*)(probe const &, interval<From, To>));
                };

                template <class Functor, class From>
                using unsafe_resolve_to = decltype(to_resolver<From>::select(&Functor::template apply<probe const &>));

                template <class Functor, class From, class = void>
                struct find_interval_parameter {
                    using type = meta::list<>;
                };

                template <class Functor, class From>
                struct find_interval_parameter<Functor, From, void_t<unsafe_resolve_to<Functor, From>>> {
                    using type = meta::list<interval<From, unsafe_resolve_to<Functor, From>>>;
                };

                template <class Functor>
                struct find_interval_parameter_f {
                    template <class Interval>
                    using apply = typename find_interval_parameter<Functor, meta::first<Interval>>::type;
                };

                template <class Interval>
                struct interval_intersects_with {
                    using from_index_t = level_to_index<meta::first<Interval>>;
                    using to_index_t = level_to_index<meta::second<Interval>>;
                    static_assert(from_index_t::value <= to_index_t::value, GT_INTERNAL_ERROR);
                    template <class Other>
                    using apply = bool_constant<level_to_index<meta::first<Other>>::value <= to_index_t::value &&
                                                level_to_index<meta::second<Other>>::value >= from_index_t::value>;
                };

                template <class Src>
                struct make_interval_from_index_f {
                    using from_index_t = level_to_index<meta::first<Src>>;
                    template <class N>
                    using make_level = index_to_level<level_index<N::value + from_index_t::value, Src::offset_limit>>;
                    template <class N>
                    using apply = interval<make_level<N>, make_level<N>>;
                };

                template <class Interval>
                using split_interval = meta::transform<make_interval_from_index_f<Interval>::template apply,
                    meta::make_indices_c<level_to_index<meta::second<Interval>>::value -
                                         level_to_index<meta::first<Interval>>::value + 1>>;

                template <class Functor, class Interval>
                using find_interval_parameters = meta::filter<interval_intersects_with<Interval>::template apply,
                    meta::flatten<meta::transform<find_interval_parameter_f<Functor>::template apply,
                        split_interval<interval<index_to_level<level_index<0, Interval::offset_limit>>,
                            meta::second<Interval>>>>>>;

                template <class F, class Lhs, class Rhs>
                    struct intersection_detector
                    : bool_constant <
                      level_to_index<meta::second<Lhs>>::value<level_to_index<meta::first<Rhs>>::value> {
                    static_assert(intersection_detector<F, Lhs, Rhs>::value,
                        "A stencil operator with intersecting intervals was detected.\nSearch above "
                        "for `intersection_detector` in this compiler error output to determine the functor and the "
                        "intervals.");
                };

                template <class Functor, class Interval>
                struct has_any_apply
                    : bool_constant<has_apply<Functor>::value ||
                                    meta::length<find_interval_parameters<Functor, Interval>>::value != 0> {
                    static_assert(has_any_apply<Functor, Interval>::value,
                        "Elementary functor doesn't have any apply overload within the given interval.\nSearch above "
                        "for `has_any_apply` in this compiler error output to determine the functor and the interval.");
                };

                template <template <class...> class F, class L>
                struct transform_neighbours;

                template <template <class...> class F, template <class...> class L>
                struct transform_neighbours<F, L<>> {
                    using type = L<>;
                };

                template <template <class...> class F, template <class...> class L, class T>
                struct transform_neighbours<F, L<T>> {
                    using type = L<>;
                };

                template <template <class...> class F, template <class...> class L, class T0, class T1, class... Ts>
                struct transform_neighbours<F, L<T0, T1, Ts...>> {
                    using type = meta::push_front<typename transform_neighbours<F, L<T1, Ts...>>::type, F<T0, T1>>;
                };

                template <class Functor, class Interval>
                struct invalid_apply_detector {
                    using intervals_t = find_interval_parameters<Functor, Interval>;
                    static constexpr bool has_any_apply =
                        has_apply<Functor>::value || meta::length<intervals_t>::value != 0;
                    static_assert(has_any_apply, "Elementary functor doesn't have apply.");
                    using intersection_detectors_t =
                        typename transform_neighbours<meta::curry<intersection_detector, Functor>::template apply,
                            intervals_t>::type;
                    using type = bool_constant<has_any_apply && meta::all<intersection_detectors_t>::value>;
                };

                // if overloads are valid this alias evaluate to std::true_type
                // otherwise static_assert is triggered.
                template <class Functor,
                    class Interval,
                    class HasAnyApply = has_any_apply<Functor, Interval>,
                    class IntersectionDetectors =
                        typename transform_neighbours<meta::curry<intersection_detector, Functor>::template apply,
                            find_interval_parameters<Functor, Interval>>::type>
                using check_valid_apply_overloads = meta::all<meta::push_back<IntersectionDetectors, HasAnyApply>>;

                template <class Index, class Intervals>
                struct find_in_interval_parameters;

                template <class Index, template <class...> class L>
                struct find_in_interval_parameters<Index, L<>> {
                    using type = meta::list<>;
                };

                template <class Index, template <class...> class L, class Interval, class... Intervals>
                struct find_in_interval_parameters<Index, L<Interval, Intervals...>> {
                    using type = typename meta::if_c<(level_to_index<meta::first<Interval>>::value > Index::value),
                        meta::list<>,
                        meta::if_c<(level_to_index<meta::second<Interval>>::value >= Index::value),
                            meta::list<Interval>,
                            find_in_interval_parameters<Index, L<Intervals...>>>>::type;
                };

                template <class Key,
                    class Functor,
                    class Interval,
                    class Params = typename find_in_interval_parameters<level_to_index<meta::first<Key>>,
                        find_interval_parameters<Functor, Interval>>::type,
                    bool HasApply = has_apply<Functor>::value>
                struct make_functor_map_item;

                template <class Key, class Functor, class Interval>
                struct make_functor_map_item<Key, Functor, Interval, meta::list<>, false> {
                    using type = meta::list<Key>;
                };

                template <class Key, class Functor, class Interval>
                struct make_functor_map_item<Key, Functor, Interval, meta::list<>, true> {
                    using type = meta::list<Key, Functor>;
                };

                template <class Functor, class Param>
                struct bound_functor : Functor {
                    template <class Eval>
                    static GT_FUNCTION void apply(Eval &&eval) {
                        Functor::apply(wstd::forward<Eval>(eval), Param());
                    }
                };

                template <class Key, class Functor, class Interval, class Param, bool HasApply>
                struct make_functor_map_item<Key, Functor, Interval, meta::list<Param>, HasApply> {
                    using type = meta::list<Key, bound_functor<Functor, Param>>;
                };

                template <class Functor, class Interval>
                struct item_maker_f {
                    template <class Key>
                    using apply = typename make_functor_map_item<Key, Functor, Interval>::type;
                };

                template <class Functor, class Interval>
                using make_functor_map =
                    meta::transform<item_maker_f<Functor, Interval>::template apply, split_interval<Interval>>;

            } // namespace functor_metafunctions_impl_
            using functor_metafunctions_impl_::check_valid_apply_overloads;
            using functor_metafunctions_impl_::invalid_apply_detector;
            using functor_metafunctions_impl_::make_functor_map;
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
