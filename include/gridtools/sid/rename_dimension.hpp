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
        namespace rename_dimension_impl_ {
            template <class OldKey, class NewKey, class Map>
            auto remap(Map map) {
                return hymap::convert_to<hymap::keys, meta::replace<get_keys<Map>, OldKey, NewKey>>(std::move(map));
            }

            template <class OldKey, class NewKey, class Sid>
            struct renamed_sid : delegate<Sid> {
                template <class Map>
                using remaped_t = decltype(remap<OldKey, NewKey>(std::declval<Map>()));

                template <class T>
                renamed_sid(T &&obj) : delegate<Sid>(std::forward<T>(obj)) {}

                friend remaped_t<strides_type<Sid>> sid_get_strides(renamed_sid const &obj) {
                    return remap<OldKey, NewKey>(sid_get_strides(obj.impl()));
                }
                friend remaped_t<lower_bounds_type<Sid>> sid_get_lower_bounds(renamed_sid const &obj) {
                    return remap<OldKey, NewKey>(sid_get_lower_bounds(obj.impl()));
                }
                friend remaped_t<upper_bounds_type<Sid>> sid_get_upper_bounds(renamed_sid const &obj) {
                    return remap<OldKey, NewKey>(sid_get_upper_bounds(obj.impl()));
                }
            };

            template <class...>
            struct stride_kind_wrapper {};

            template <class OldKey, class NewKey, class Sid>
            stride_kind_wrapper<OldKey, NewKey, strides_kind<Sid>> sid_get_strides_kind(
                renamed_sid<OldKey, NewKey, Sid> const &);

            template <class OldKey, class NewKey, class Sid>
            renamed_sid<OldKey, NewKey, Sid> rename_dimension(Sid &&sid) {
                return renamed_sid<OldKey, NewKey, Sid>{std::forward<Sid>(sid)};
            }
        } // namespace rename_dimension_impl_
        using rename_dimension_impl_::rename_dimension;
    } // namespace sid
} // namespace gridtools
