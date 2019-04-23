/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef GT_TARGET_ITERATING
// DON'T USE #pragma once HERE!!!
#ifndef GT_STENCIL_COMPOSITION_SID_SIMPLE_PTR_HOLDER_HPP_
#define GT_STENCIL_COMPOSITION_SID_SIMPLE_PTR_HOLDER_HPP_

#include <utility>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"
#include "../../meta/utility.hpp"

#define GT_FILENAME <gridtools/stencil_composition/sid/simple_ptr_holder.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif
#else

namespace gridtools {
    namespace sid {
        GT_TARGET_NAMESPACE {
            template <class T>
            struct simple_ptr_holder {
                T m_val;
                GT_TARGET GT_FORCE_INLINE GT_HOST_CONSTEXPR T const &operator()() const { return m_val; }
            };

            template <class T>
            GT_FORCE_INLINE constexpr simple_ptr_holder<T> make_simple_ptr_holder(T const &ptr) {
                return {ptr};
            }

            template <class T, class Arg>
            GT_FORCE_INLINE constexpr auto operator+(simple_ptr_holder<T> const &obj, Arg &&arg)
                GT_AUTO_RETURN(make_simple_ptr_holder(obj.m_val + const_expr::forward<Arg>(arg)));

            template <class T, class Arg>
            GT_FORCE_INLINE constexpr auto operator+(simple_ptr_holder<T> &&obj, Arg &&arg)
                GT_AUTO_RETURN(make_simple_ptr_holder(const_expr::move(obj).m_val + const_expr::forward<Arg>(arg)));
        }
    } // namespace sid
} // namespace gridtools

#endif
