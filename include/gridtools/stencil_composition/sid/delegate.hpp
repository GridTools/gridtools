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

            GT_STATIC_ASSERT(is_sid<Sid>::value, GT_INTERNAL_ERROR);

            // Clang does not like GT_AUTO_RETURN here
            friend GT_CONSTEXPR decltype(get_origin(std::declval<Sid &>())) sid_get_origin(delegate &obj) {
                return get_origin(obj.m_impl);
            }
            friend GT_CONSTEXPR decltype(get_strides(std::declval<Sid const &>())) sid_get_strides(
                delegate const &obj) {
                return get_strides(obj.m_impl);
            }

          protected:
            GT_CONSTEXPR Sid const &impl() const { return m_impl; }
            Sid &impl() { return m_impl; }

          public:
            template <class Arg>
            explicit GT_CONSTEXPR delegate(std::initializer_list<Arg> lst) : m_impl(*lst.begin()) {}

            template <class Arg>
            explicit GT_CONSTEXPR delegate(Arg &&arg) noexcept : m_impl(wstd::forward<Arg>(arg)) {}
        };

        template <class Sid>
        GT_META_CALL(ptr_diff_type, Sid)
        sid_get_ptr_diff(delegate<Sid> const &);

        template <class Sid>
        GT_META_CALL(strides_kind, Sid)
        sid_get_strides_kind(delegate<Sid> const &);
    } // namespace sid
} // namespace gridtools
