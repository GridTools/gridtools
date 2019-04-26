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
 *  - `get_keys` metafunction. Usage: `GT_META_CALL(get_keys, Hymap)`
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
        GT_META_DEFINE_ALIAS(get_key, meta::id, (integral_constant<int, I::value>));

        template <class T>
        GT_META_DEFINE_ALIAS(default_keys,
            meta::transform,
            (get_key, GT_META_CALL(meta::make_indices_for, GT_META_CALL(tuple_util::traits::to_types, T))));

        struct not_provided;

        not_provided hymap_get_keys(...);

        template <class T, class Res = decltype(hymap_get_keys(std::declval<T const &>()))>
        enable_if_t<!std::is_same<Res, not_provided>::value, Res> get_keys_fun(T const &);

        template <class T, class Res = decltype(hymap_get_keys(std::declval<T const &>()))>
        enable_if_t<std::is_same<Res, not_provided>::value, GT_META_CALL(default_keys, T)> get_keys_fun(T const &);

        template <class T>
        using get_keys = decltype(::gridtools::hymap_impl_::get_keys_fun(std::declval<T const &>()));
    } // namespace hymap_impl_

#if GT_BROKEN_TEMPLATE_ALIASES
    template <class T>
    struct get_keys : meta::id<hymap_impl_::get_keys<T>> {};
#else
    using hymap_impl_::get_keys;
#endif

    template <class Map, class Key>
    GT_META_DEFINE_ALIAS(has_key, meta::st_contains, (GT_META_CALL(hymap_impl_::get_keys, Map), Key));

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

                using type = values;
            };
            using type = keys;
        };

        template <class HyMap>
        GT_META_DEFINE_ALIAS(to_meta_map,
            meta::zip,
            (GT_META_CALL(meta::rename, (meta::list, GT_META_CALL(get_keys, HyMap))),
                GT_META_CALL(meta::rename, (meta::list, GT_META_CALL(tuple_util::traits::to_types, HyMap)))));

        template <class MetaMap>
        GT_META_DEFINE_ALIAS(from_meta_map,
            meta::rename,
            (GT_META_CALL(
                 meta::rename, (hymap::keys, GT_META_CALL(meta::transform, (meta::first, MetaMap))))::template values,
                GT_META_CALL(meta::transform, (meta::second, MetaMap))));

        template <class HyMap,
            class Key,
            class I = GT_META_CALL(meta::st_position, (GT_META_CALL(get_keys, HyMap), Key))>
        GT_META_DEFINE_ALIAS(value_type_at_key, meta::at, (GT_META_CALL(tuple_util::traits::to_types, HyMap), I));

    } // namespace hymap
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
            class I = GT_META_CALL(meta::st_position, (GT_META_CALL(get_keys, decay_t<Map>), Key))>
        GT_TARGET GT_FORCE_INLINE constexpr auto at_key(Map && map) noexcept GT_AUTO_RETURN(
            tuple_util::GT_TARGET_NAMESPACE_NAME::get<I::value>(const_expr::forward<Map>(map)));

        template <class Key,
            class Default,
            class Map,
            class Decayed = decay_t<Map>,
            class I = GT_META_CALL(meta::st_position, (GT_META_CALL(get_keys, Decayed), Key)),
            enable_if_t<I::value != tuple_util::size<Decayed>::value, int> = 0>
        GT_TARGET GT_FORCE_INLINE constexpr auto at_key_with_default(Map && map) noexcept GT_AUTO_RETURN(
            tuple_util::GT_TARGET_NAMESPACE_NAME::get<I::value>(const_expr::forward<Map>(map)));

        template <class Key,
            class Default,
            class Map,
            class Decayed = decay_t<Map>,
            class I = GT_META_CALL(meta::st_position, (GT_META_CALL(get_keys, Decayed), Key)),
            enable_if_t<I::value == tuple_util::size<Decayed>::value, int> = 0>
        GT_TARGET GT_FORCE_INLINE constexpr Default at_key_with_default(Map &&) noexcept {
            return {};
        }
    }
} // namespace gridtools

#endif // GT_TARGET_ITERATING
