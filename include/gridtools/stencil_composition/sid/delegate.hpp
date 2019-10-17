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

#include "../../common/defs.hpp"
#include "../../meta/macros.hpp"
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
            friend strides_type<Sid> sid_get_strides(delegate const &obj) { return get_strides(obj.m_impl); }
            friend lower_bounds_type<Sid> sid_get_lower_bounds(delegate const &obj) {
                return get_lower_bounds(obj.m_impl);
            }
            friend upper_bounds_type<Sid> sid_get_upper_bounds(delegate const &obj) {
                return get_upper_bounds(obj.m_impl);
            }

          protected:
            Sid const &impl() const { return m_impl; }
            Sid &impl() { return m_impl; }

          public:
            template <class Arg>
            explicit delegate(std::initializer_list<Arg> lst) : m_impl(*lst.begin()) {}

            template <class Arg>
            explicit delegate(Arg &&arg) noexcept : m_impl(std::forward<Arg>(arg)) {}
        };

        template <class Sid>
        ptr_diff_type<Sid> sid_get_ptr_diff(delegate<Sid> const &);

        template <class Sid>
        strides_kind<Sid> sid_get_strides_kind(delegate<Sid> const &);
    } // namespace sid
} // namespace gridtools
