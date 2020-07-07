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
#include <utility>

#include "../common/defs.hpp"
#include "../meta/macros.hpp"
#include "concept.hpp"

namespace gridtools {
    namespace sid {
        /**
         *  A helper class for implementing delegate design pattern for `SID`s
         *  Typically the user template class should inherit from `delegate`
         *  For example please look into `test_sid_delegate.cpp`
         *
         * @tparam Sid a object that models `SID` concept.
         */
        template <class Sid>
        struct delegate {
            static_assert(is_sid<Sid>::value, GT_INTERNAL_ERROR);
            Sid m_impl;

            template <bool IsRef = std::is_reference<Sid>::value, std::enable_if_t<!IsRef, int> = 0>
            delegate(Sid impl) : m_impl(std::move(impl)) {}

            template <bool IsRef = std::is_reference<Sid>::value, std::enable_if_t<IsRef, int> = 0>
            delegate(Sid impl) : m_impl(impl) {}
        };

        template <class Sid>
        decltype(sid_get_origin(std::declval<Sid &>())) sid_get_origin(delegate<Sid> &obj) {
            return sid_get_origin(obj.m_impl);
        }

        template <class Sid>
        decltype(sid_get_ptr_diff(std::declval<Sid const &>())) sid_get_ptr_diff(delegate<Sid> const &);

        template <class Sid>
        decltype(sid_get_strides_kind(std::declval<Sid const &>())) sid_get_strides_kind(delegate<Sid> const &);

        template <class Sid>
        decltype(sid_get_strides(std::declval<Sid const &>())) sid_get_strides(delegate<Sid> const &obj) {
            return sid_get_strides(obj.m_impl);
        }
        template <class Sid>
        decltype(sid_get_lower_bounds(std::declval<Sid const &>())) sid_get_lower_bounds(delegate<Sid> const &obj) {
            return sid_get_lower_bounds(obj.m_impl);
        }
        template <class Sid>
        decltype(sid_get_upper_bounds(std::declval<Sid const &>())) sid_get_upper_bounds(delegate<Sid> const &obj) {
            return sid_get_upper_bounds(obj.m_impl);
        }
    } // namespace sid
} // namespace gridtools
