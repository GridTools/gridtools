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

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/utility.hpp"
#include "../../meta/type_traits.hpp"

namespace gridtools {
    namespace sid {

        enum class property { origin, strides, ptr_diff, strides_kind };

        namespace synthetic_impl_ {
            template <property Property, class T>
            struct mixin;

            template <class T>
            struct mixin<property::origin, T> {
                T m_val;
            };
            template <class T>
            T sid_get_origin(mixin<property::origin, T> const &obj) noexcept {
                return obj.m_val;
            }

            template <class T>
            struct mixin<property::strides, T> {
                T m_val;
            };
            template <class T>
            T sid_get_strides(mixin<property::strides, T> const &obj) noexcept {
                return obj.m_val;
            }

            template <class T>
            struct mixin<property::ptr_diff, T> {};
            template <class T>
            T sid_get_ptr_diff(mixin<property::ptr_diff, T> const &obj);

            template <class T>
            struct mixin<property::strides_kind, T> {};
            template <class T>
            T sid_get_strides_kind(mixin<property::strides_kind, T> const &);

            template <property>
            struct unique {};

            template <property Property, class T>
            struct unique_mixin : mixin<Property, T>, unique<Property> {
                GT_DECLARE_DEFAULT_EMPTY_CTOR(unique_mixin);
                unique_mixin(unique_mixin const &) = default;
                unique_mixin(unique_mixin &&) = default;
                unique_mixin &operator=(unique_mixin const &) = default;
                unique_mixin &operator=(unique_mixin &&) = default;

                template <class U>
                unique_mixin(U &&obj) noexcept : mixin<Property, T>{const_expr::forward<U>(obj)} {}
            };

            template <class...>
            struct synthetic;

            template <>
            struct synthetic<> {
                template <property Property, class T>
                synthetic<unique_mixin<Property, T>> set() const &&noexcept {
                    return synthetic{};
                }

                template <property Property, class T>
                synthetic<unique_mixin<Property, decay_t<T>>> set(T &&val) const &&noexcept {
                    return {const_expr::forward<T>(val), synthetic{}};
                }
            };

            template <class Mixin, class... Mixins>
            struct synthetic<Mixin, Mixins...> : Mixin, Mixins... {

                GT_DECLARE_DEFAULT_EMPTY_CTOR(synthetic);

                synthetic(synthetic<Mixins...> const &&src) noexcept : Mixins(const_expr::move(src))... {}

                template <class T>
                synthetic(T &&val, synthetic<Mixins...> const &&src) noexcept
                    : Mixin{const_expr::forward<T>(val)}, Mixins(const_expr::move(src))... {}

                template <property Property, class U>
                synthetic<unique_mixin<Property, U>, Mixin, Mixins...> set() const &&noexcept {
                    return {const_expr::move(*this)};
                }

                template <property Property, class T>
                synthetic<unique_mixin<Property, decay_t<T>>, Mixin, Mixins...> set(T &&val) const &&noexcept {
                    return {const_expr::forward<T>(val), const_expr::move(*this)};
                }
            };
        } // namespace synthetic_impl_

        /**
         *  A tiny EDSL for creating SIDs from the parts described in the concept.
         *
         *  Usage:
         *
         *  \code
         *  auto my_sid = synthetic()
         *      .set<property::origin>(origin)
         *      .set<property::strides>(strides)
         *      .set<property::ptr_diff, ptr_diff>()
         *      .set<property::strides_kind, strides_kind>();
         *  \endcode
         *
         *  only `set<property::origin>` is required. Other `set`'s can be skipped.
         *  `set`'s can go in any order. `set` of the given property can participate at most once.
         *  Duplicated property `set`'s cause compiler error.
         *  Works both in run time and in compile time.
         */
        synthetic_impl_::synthetic<> synthetic() { return {}; }
    } // namespace sid
} // namespace gridtools
