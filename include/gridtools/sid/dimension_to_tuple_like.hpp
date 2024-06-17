/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <type_traits>
#include <utility>

#include "../common/host_device.hpp"
#include "../common/hymap.hpp"
#include "../meta.hpp"
#include "composite.hpp"
#include "concept.hpp"
#include "delegate.hpp"
#include "sid_shift_origin.hpp"

namespace gridtools {
    namespace sid {
        namespace dimension_to_tuple_like_impl_ {
            template <class Dim, class Sid>
            struct remove_dimension_sid : sid::delegate<Sid> {
                friend decltype(hymap::canonicalize_and_remove_key<Dim>(std::declval<sid::strides_type<Sid>>()))
                sid_get_strides(remove_dimension_sid const &obj) {
                    return hymap::canonicalize_and_remove_key<Dim>(sid::get_strides(obj.m_impl));
                }
                friend decltype(hymap::canonicalize_and_remove_key<Dim>(std::declval<sid::lower_bounds_type<Sid>>()))
                sid_get_lower_bounds(remove_dimension_sid const &obj) {
                    return hymap::canonicalize_and_remove_key<Dim>(sid::get_lower_bounds(obj.m_impl));
                }
                friend decltype(hymap::canonicalize_and_remove_key<Dim>(std::declval<sid::upper_bounds_type<Sid>>()))
                sid_get_upper_bounds(remove_dimension_sid const &obj) {
                    return hymap::canonicalize_and_remove_key<Dim>(sid::get_upper_bounds(obj.m_impl));
                }

                using sid::delegate<Sid>::delegate;
            };

            template <class Dim, class Sid>
            remove_dimension_sid<Dim, Sid> remove_dimension(Sid &&sid) {
                return {std::forward<Sid>(sid)};
            }

            template <class T, class U>
            std::enable_if_t<std::is_reference_v<T>, T> copy_if_rvalue(U &p) {
                return p;
            }
            template <class T, class U>
            std::enable_if_t<!std::is_reference_v<T>, T> copy_if_rvalue(U const &p) {
                return p;
            }

            template <class Dim, class Sid, size_t... Is>
            constexpr decltype(auto) as_tuple_like_helper(Sid &&sid, std::index_sequence<Is...>) {
                using keys = sid::composite::keys<integral_constant<int, Is>...>;
                return keys::make_values(remove_dimension<Dim>(sid::shift_sid_origin(
                    copy_if_rvalue<Sid>(sid), // copy required otherwise we move away from `sid` multiple times
                    hymap::keys<Dim>::make_values(Is)))...);
            }
        } // namespace dimension_to_tuple_like_impl_

        /**
         * Returns a SID, where `Dim` of `sid` is mapped to a tuple-like of size `N`.
         *
         * TODO(havogt):
         * - Currently, no bounds check is implemented.
         * - In case bounds are compile-time known we could infer `N`.
         */
        template <class Dim, size_t N, class Sid>
        decltype(auto) dimension_to_tuple_like(Sid &&sid) {
            return dimension_to_tuple_like_impl_::as_tuple_like_helper<Dim>(
                std::forward<Sid>(sid), std::make_index_sequence<N>{});
        }
    } // namespace sid
} // namespace gridtools
