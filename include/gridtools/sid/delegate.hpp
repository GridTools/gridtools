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

#include <initializer_list>

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
        class delegate {
            Sid m_impl;

            static_assert(is_sid<Sid>::value, GT_INTERNAL_ERROR);

            friend ptr_holder_type<Sid> sid_get_origin(delegate &obj) { return get_origin(obj.m_impl); }

          public:
            template <class Arg>
            explicit delegate(std::initializer_list<Arg> lst) : m_impl(*lst.begin()) {}

            template <class Arg>
            explicit delegate(Arg &&arg) noexcept : m_impl(std::forward<Arg>(arg)) {}

            Sid const &impl() const { return m_impl; }
            Sid &impl() { return m_impl; }
        };

        template <class Sid>
        decltype(sid_get_ptr_diff(std::declval<Sid const &>())) sid_get_ptr_diff(delegate<Sid> const &);

        template <class Sid>
        decltype(sid_get_strides_kind(std::declval<Sid const &>())) sid_get_strides_kind(delegate<Sid> const &);

        template <class Sid>
        decltype(sid_get_strides(std::declval<Sid const &>())) sid_get_strides(delegate<Sid> const &obj) {
            return sid_get_strides(obj.impl());
        }
        template <class Sid>
        decltype(sid_get_lower_bounds(std::declval<Sid const &>())) sid_get_lower_bounds(delegate<Sid> const &obj) {
            return sid_get_lower_bounds(obj.impl());
        }
        template <class Sid>
        decltype(sid_get_upper_bounds(std::declval<Sid const &>())) sid_get_upper_bounds(delegate<Sid> const &obj) {
            return sid_get_upper_bounds(obj.impl());
        }
    } // namespace sid
} // namespace gridtools
