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

                template <class Intervals>
                struct are_intervals_valid;

                template <template <class...> class L>
                struct are_intervals_valid<L<>> : std::true_type {};

                template <template <class...> class L, class Interval>
                struct are_intervals_valid<L<Interval>> : std::true_type {};

                template <template <class...> class L, class First, class Second, class... Intervals>
                struct are_intervals_valid<L<First, Second, Intervals...>>
                    : bool_constant<(level_to_index<meta::second<First>>::value <
                                        level_to_index<meta::first<Second>>::value) &&
                                    are_intervals_valid<meta::list<Second, Intervals...>>::value> {};

                template <class Functor,
                    class Interval,
                    class Params = find_interval_parameters<Functor, Interval>,
                    class HasApply = has_apply<Functor>>
                using is_valid_functor = bool_constant<are_intervals_valid<Params>::value &&
                                                       (HasApply::value || meta::length<Params>::value != 0)>;

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
            using functor_metafunctions_impl_::is_valid_functor;
            using functor_metafunctions_impl_::make_functor_map;
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
