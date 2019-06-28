/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 *  @file
 *
 *  Hybrid Map (aka hymap) Concept Definition
 *  -----------------------------------------
 *  Hymap is a `tuple_like` (see tuple_util.hpp for definition) that additionally holds a type list of keys.
 *  This type list should be a set. The existense of it allows to use an alternative syntax for element accessor
 *  (in addition to `tuple_util::get`). Say you have a hymap `obj` that is a `tuple_like` of two elements and has
 *  keys `a` and `b`. To access the elements you can use: `at_key<a>(obj)` and `at_key<b>(obj)` which are semantically
 *  the same as `get<0>(obj)` and `get<1>(obj)`.
 *
 *  Regarding the Naming
 *  --------------------
 *  Hymaps provides a mechanism for mapping compile time keys to run time values. That is why it is hybrid map.
 *  Keys exist only in compile time -- it is enough to have just a type list to keep them; values are kept in the
 *  `tuple_like` -- that is why hymap also models `tuple_like` for its values.
 *
 *  Concept modeler API
 *  -------------------
 *  Hymaps provide the mentioned keys type lists by declaring the function that should be available by ADL:
 *  `Keys hymap_get_keys(Hymap)`.
 *
 *  The Default Behaviour
 *  ---------------------
 *  If hymap doesn't provide `hymap_get_keys`, the default is taken which is:
 *  `meta::list<integral_constant<int, 0>, ..., integral_constant<int, N> >` where N is `tuple_util::size` of hymap.
 *  This means that any `tuple_like` automatically models Hymap. And for the plain `tuple_like`'s you can use
 *  `at_key<integral_constant<int, N>>(obj)` instead of `tuple_util::get<N>(obj)`.
 *
 *  User API
 *  --------
 *  Run time: a single function `target_name::at_key<Key>(hymap_obj)` is provided where `target_name` is `host`,
 * `device` or `host_device`. `at_key` without `target_name` is an alias of `host::at_key`.
 *
 *  Compile time:
 *  - `get_keys` metafunction. Usage: `get_keys<Hymap>`
 *  - `has_key` metafunction. Usage `has_key<Hymap, Key>`
 *
 *  TODO(anstaf): add usage examples here
 *
 *  Gridtools implementation of Hymap
 *  ---------------------------------
 *
 *  Usage
 *  -----
 *  ```
 *    struct a;
 *    struct b;
 *    using my_map_t = hymap::keys<a, b>::values<int, double>;
 *  ```
 *
 *  Composing with `tuple_util` library
 *  -----------------------------------
 *  Because `hymap` is also a `tuple_like`, all `tuple_util` stuff is at your service.
 *  For example:
 *  - transforming the values of the hymap:
 *    `auto dst_hymap = tuple_util::transform(change_value_functor, src_hymap);`
 *  - making a map:
 *    `auto a_map = tuple_util::make<hymap::keys<a, b>::values>('a', 42)`
 *  - converting to a map:
 *    `auto a_map = tuple_util::convert_to<hymap::keys<a, b>::values>(a_tuple_with_values)`
 */

#ifndef GT_TARGET_ITERATING
//// DON'T USE #pragma once HERE!!!
#ifndef GT_COMMON_HYMAP_HPP_
#define GT_COMMON_HYMAP_HPP_

#include <type_traits>

#include "../meta.hpp"
#include "defs.hpp"
#include "generic_metafunctions/utility.hpp"
#include "host_device.hpp"
#include "integral_constant.hpp"
#include "tuple.hpp"
#include "tuple_util.hpp"

namespace gridtools {

    namespace hymap_impl_ {

        template <class I>
        using get_key = integral_constant<int, I::value>;

        template <class Tup, class Ts = tuple_util::traits::to_types<Tup>>
        using default_keys = meta::transform<get_key, meta::make_indices_for<Ts>>;

        struct not_provided;

        not_provided hymap_get_keys(...);

        template <class T, class Res = decltype(hymap_get_keys(std::declval<T const &>()))>
        std::enable_if_t<!std::is_same<Res, not_provided>::value, Res> get_keys_fun(T const &);

        template <class T, class Res = decltype(hymap_get_keys(std::declval<T const &>()))>
        std::enable_if_t<std::is_same<Res, not_provided>::value, default_keys<T>> get_keys_fun(T const &);

        template <class T>
        using get_keys = decltype(::gridtools::hymap_impl_::get_keys_fun(std::declval<T const &>()));
    } // namespace hymap_impl_

    using hymap_impl_::get_keys;

    template <class Map, class Key>
    using has_key = meta::st_contains<hymap_impl_::get_keys<Map>, Key>;

    namespace hymap {
        template <class... Keys>
        struct keys {
            template <class... Vals>
            struct values {
                GT_STATIC_ASSERT(sizeof...(Vals) == sizeof...(Keys), "invalid hymap");

                tuple<Vals...> m_vals;

                GT_TUPLE_UTIL_FORWARD_CTORS_TO_MEMBER(values, m_vals);
                GT_TUPLE_UTIL_FORWARD_GETTER_TO_MEMBER(values, m_vals);

                friend keys hymap_get_keys(values const &) { return {}; }
            };
        };

        template <class HyMap>
        using to_meta_map = meta::zip<get_keys<HyMap>, tuple_util::traits::to_types<HyMap>>;

        template <class Keys, class Values, class HyMapKeys = meta::rename<keys, Keys>>
        using from_keys_values = meta::rename<HyMapKeys::template values, Values>;

        template <class MetaMap,
            class KeysAndValues = meta::transpose<MetaMap>,
            class Keys = meta::first<KeysAndValues>,
            class Values = meta::second<KeysAndValues>>
        using from_meta_map = from_keys_values<Keys, Values>;

        namespace hymap_impl_ {
            template <class Maps>
            using merged_keys = meta::dedup<meta::transform<meta::first, meta::flatten<Maps>>>;

            template <class Key>
            struct find_f {
                template <class Map>
                using apply = meta::second<meta::mp_find<Map, Key, meta::list<void, void>>>;
            };

            template <class State, class Val>
            using get_first_folder = meta::if_<std::is_void<State>, Val, State>;

            template <class Maps>
            struct merged_value_f {
                template <class Key>
                using apply = meta::lfold<get_first_folder, void, meta::transform<find_f<Key>::template apply, Maps>>;
            };

            template <class Src>
            using map_of_refs = decltype(tuple_util::transform(identity{}, std::declval<Src>()));

            template <class Maps,
                class RefMaps = meta::transform<map_of_refs, Maps>,
                class MetaMaps = meta::transform<to_meta_map, RefMaps>,
                class Keys = merged_keys<MetaMaps>,
                class Values = meta::transform<merged_value_f<MetaMaps>::template apply, Keys>>
            using merged_old = from_keys_values<Keys, Values>;
        } // namespace hymap_impl_
    }     // namespace hymap
} // namespace gridtools

#define GT_FILENAME <gridtools/common/hymap.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif // GT_COMMON_HYMAP_HPP_
#else  // GT_TARGET_ITERATING

namespace gridtools {
    GT_TARGET_NAMESPACE {
        template <class Key,
            class Map,
            class... Maps,
            class Decayed = std::decay_t<Map>,
            class I = meta::st_position<get_keys<Decayed>, Key>,
            std::enable_if_t<I::value != tuple_util::size<Decayed>::value, int> = 0>
        GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR decltype(auto) at_key(Map && map, Maps && ...) noexcept {
            return tuple_util::GT_TARGET_NAMESPACE_NAME::get<I::value>(wstd::forward<Map>(map));
        }

        template <class Key>
        GT_TARGET void at_key() {
            GT_STATIC_ASSERT(sizeof(Key) != sizeof(Key), "wrong key");
        }

        template <class Key,
            class Map,
            class... Maps,
            class Decayed = std::decay_t<Map>,
            class I = meta::st_position<get_keys<Decayed>, Key>,
            std::enable_if_t<I::value == tuple_util::size<Decayed>::value, int> = 0>
        GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR decltype(auto) at_key(Map && map, Maps && ... maps) noexcept {
            return GT_TARGET_NAMESPACE_NAME::at_key<Key>(wstd::forward<Maps>(maps)...);
        }

        template <class Key,
            class Default,
            class Map,
            class Decayed = std::decay_t<Map>,
            class I = meta::st_position<get_keys<Decayed>, Key>,
            std::enable_if_t<I::value != tuple_util::size<Decayed>::value, int> = 0>
        GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR decltype(auto) at_key_with_default(Map && map) noexcept {
            return tuple_util::GT_TARGET_NAMESPACE_NAME::get<I::value>(wstd::forward<Map>(map));
        }

        template <class Key,
            class Default,
            class Map,
            class Decayed = std::decay_t<Map>,
            class I = meta::st_position<get_keys<Decayed>, Key>,
            std::enable_if_t<I::value == tuple_util::size<Decayed>::value, int> = 0>
        GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Default at_key_with_default(Map &&) noexcept {
            return {};
        }
    }

    namespace hymap {
        GT_TARGET_NAMESPACE {
            namespace hymap_detail {
                template <class Fun, class Keys>
                struct adapter_f {
                    Fun m_fun;
                    template <size_t I, class Value, class Key = meta::at_c<Keys, I>>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR decltype(auto) operator()(Value &&value) const {
                        return m_fun.template operator()<Key>(wstd::forward<Value>(value));
                    }
                };

                template <class Key>
                struct merged_generator_f {
                    template <class... Maps>
                    GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR decltype(auto) operator()(Maps &&... maps) const {
                        return gridtools::GT_TARGET_NAMESPACE_NAME::at_key<Key>(wstd::forward<Maps>(maps)...);
                    }
                };
            } // namespace hymap_detail

            template <class Fun, class Map>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto transform(Fun && fun, Map && map) {
                return tuple_util::GT_TARGET_NAMESPACE_NAME::transform_index(
                    hymap_detail::adapter_f<Fun, get_keys<std::decay_t<Map>>>{wstd::forward<Fun>(fun)},
                    wstd::forward<Map>(map));
            }

            template <class Fun, class Map>
            GT_TARGET GT_FORCE_INLINE void for_each(Fun && fun, Map && map) {
                tuple_util::GT_TARGET_NAMESPACE_NAME::for_each_index(
                    hymap_detail::adapter_f<Fun, get_keys<std::decay_t<Map>>>{wstd::forward<Fun>(fun)},
                    wstd::forward<Map>(map));
            }

            template <class Primary, class Secondary>
            class merged : tuple<Primary, Secondary> {
                using base_t = tuple<Primary, Secondary>;

                using meta_primary_t = to_meta_map<Primary>;
                using meta_secondary_t = to_meta_map<Secondary>;

                using primary_keys_t = meta::transform<meta::first, meta_primary_t>;
                using secondary_keys_t = meta::transform<meta::first, meta_secondary_t>;

                template <class Key,
                    class PrimaryItem = meta::mp_find<meta_primary_t, Key>,
                    class SecondaryItem = meta::mp_find<meta_secondary_t, Key>>
                using get_value_type = meta::second<meta::if_<std::is_void<PrimaryItem>, SecondaryItem, PrimaryItem>>;

                //                template <class Key, class Map = meta::st_contains<primary_keys_t, Key>, class Index =
                //                void> using get_index_map_item = meta::list<Key, Map, Index>;

                using keys_t = meta::dedup<meta::concat<primary_keys_t, secondary_keys_t>>;
                using values_t = meta::transform<get_value_type, keys_t>;

                template <size_t I, class Key = meta::at_c<keys_t, I>>
                using is_primary_index = meta::st_contains<primary_keys_t, Key>;

                template <size_t I, class Key = meta::at_c<keys_t, I>>
                using inner_index = meta::if_<is_primary_index<I>,
                    meta::st_position<primary_keys_t, Key>,
                    meta::st_position<secondary_keys_t, Key>>;

                template <size_t I,
                    class Key = meta::at_c<keys_t, I>,
                    class PrimaryPos = meta::st_position<primary_keys_t, Key>,
                    class SecondaryPos = meta::st_position<secondary_keys_t, Key>>
                using split_index = meta::if_c<(PrimaryPos::value < meta::length<primary_keys_t>::value),
                    meta::list<std::integral_constant<size_t, 0>, PrimaryPos>,
                    meta::list<std::integral_constant<size_t, 1>, SecondaryPos>>;

                friend values_t tuple_to_types(merged const &) { return {}; }

                friend keys_t hymap_get_keys(merged const &) { return {}; }

                friend struct merged_getter;

                GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR base_t const &base() const { return *this; }
                GT_TARGET GT_FORCE_INLINE base_t &base() { return *this; }

                template <size_t I>
                GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR decltype(auto) source() const {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get < is_primary_index<I>::value ? 0 : 1 > (base());
                }
                template <size_t I>
                GT_TARGET GT_FORCE_INLINE decltype(auto) source() {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get < is_primary_index<I>::value ? 0 : 1 > (base());
                }

                template <size_t I>
                GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR decltype(auto) get() const {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<inner_index<I>::value>(source<I>());
                }
                template <size_t I>
                GT_TARGET GT_FORCE_INLINE decltype(auto) get() {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<inner_index<I>::value>(source<I>());
                }

              public:
                merged() = default;

                GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR merged(Primary primary, Secondary secondary)
                    : base_t(wstd::move(primary), wstd::move(secondary)) {}

                GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR Primary const &primary() const {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<0>(base());
                }
                GT_TARGET GT_FORCE_INLINE Primary &primary() {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<0>(base());
                }

                GT_TARGET GT_FORCE_INLINE Secondary const &secondary() const {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<1>(base());
                }
                GT_TARGET GT_FORCE_INLINE Secondary &secondary() {
                    return tuple_util::GT_TARGET_NAMESPACE_NAME::get<1>(base());
                }

                // TODO:
                // - element wise ctor
                // - tuple_from_types
            };

            struct merged_getter {
                template <size_t I, class Primary, class Secondary>
                static GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR decltype(auto) get(
                    merged<Primary, Secondary> const &obj) {
                    return obj.template get<I>();
                }
                template <size_t I, class Primary, class Secondary>
                static GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR decltype(auto) get(merged<Primary, Secondary> &obj) {
                    return obj.template get<I>();
                }
            };

            template <class Primary, class Secondary>
            merged_getter tuple_getter(merged<Primary, Secondary> const &);

            template <class Primary, class Secondary>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR merged<Primary, Secondary> merge(
                Primary primary, Secondary secondary) {
                return {wstd::move(primary), wstd::move(secondary)};
            }

            template <class Primary, class... Secondaries>
            GT_TARGET GT_FORCE_INLINE GT_CONSTEXPR auto merge(Primary primary, Secondaries... secondaries) {
                return merge(wstd::move(primary), merge(wstd::move(secondaries)...));
            }
        } // namespace hymap
    }     // namespace hymap
} // namespace gridtools

#endif // GT_TARGET_ITERATING
