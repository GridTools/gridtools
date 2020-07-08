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

#include <utility>

#include "../common/hymap.hpp"
#include "../meta.hpp"
#include "concept.hpp"
#include "delegate.hpp"

namespace gridtools {
    namespace sid {
        namespace rename_dimensions_impl_ {

            template <class...>
            struct make_key_map;

            template <>
            struct make_key_map<> : meta::list<> {};

            template <class Old, class New, class... Keys>
            struct make_key_map<Old, New, Keys...> {
                using type = meta::push_front<typename make_key_map<Keys...>::type, meta::list<Old, New>>;
            };

            template <class KeyMap>
            struct get_new_key_f {
                template <class Key>
                using apply = meta::second<meta::mp_find<KeyMap, Key, meta::list<Key, Key>>>;
            };

            template <class KeyMap, class Map>
            auto remap(Map map) {
                return hymap::convert_to<hymap::keys,
                    meta::transform<get_new_key_f<KeyMap>::template apply, get_keys<Map>>>(std::move(map));
            }

            template <class KeyMap, class Sid>
            struct renamed_sid : delegate<Sid> {
                using delegate<Sid>::delegate;
            };

            template <class...>
            struct stride_kind_wrapper {};

            template <class KeyMap, class Sid>
            stride_kind_wrapper<KeyMap, decltype(sid_get_strides_kind(std::declval<Sid const &>()))>
            sid_get_strides_kind(renamed_sid<KeyMap, Sid> const &);

            template <class KeyMap, class Sid>
            decltype(remap<KeyMap>(sid_get_strides(std::declval<Sid const &>()))) sid_get_strides(
                renamed_sid<KeyMap, Sid> const &obj) {
                return remap<KeyMap>(sid_get_strides(obj.impl()));
            }

            template <class KeyMap, class Sid>
            decltype(remap<KeyMap>(sid_get_lower_bounds(std::declval<Sid const &>()))) sid_get_lower_bounds(
                renamed_sid<KeyMap, Sid> const &obj) {
                return remap<KeyMap>(sid_get_lower_bounds(obj.impl()));
            }

            template <class KeyMap, class Sid>
            decltype(remap<KeyMap>(sid_get_upper_bounds(std::declval<Sid const &>()))) sid_get_upper_bounds(
                renamed_sid<KeyMap, Sid> const &obj) {
                return remap<KeyMap>(sid_get_upper_bounds(obj.impl()));
            }

            template <class KeyMap, class Arr, std::enable_if_t<std::is_array<Arr>::value, int> = 0>
            auto sid_get_strides(renamed_sid<KeyMap, Arr &> const &obj) {
                return remap<KeyMap>(get_strides(obj.m_impl));
            }

            template <class KeyMap, class Arr, std::enable_if_t<std::is_array<Arr>::value, int> = 0>
            auto sid_get_lower_bounds(renamed_sid<KeyMap, Arr &> const &obj) {
                return remap<KeyMap>(get_lower_bounds(obj.m_impl));
            }

            template <class KeyMap, class Arr, std::enable_if_t<std::is_array<Arr>::value, int> = 0>
            auto sid_get_upper_bounds(renamed_sid<KeyMap, Arr &> const &obj) {
                return remap<KeyMap>(get_upper_bounds(obj.m_impl));
            }

            template <class Sid, class... Keys>
            renamed_sid<typename make_key_map<Keys...>::type, Sid> rename_dimensions(Sid &&sid, Keys...) {
                return {std::forward<Sid>(sid)};
            }
        } // namespace rename_dimensions_impl_
        using rename_dimensions_impl_::rename_dimensions;
    } // namespace sid
} // namespace gridtools
