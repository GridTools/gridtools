/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include "../../common/generic_metafunctions/utility.hpp"
#include "../../common/host_device.hpp"
#include "../../meta/type_traits.hpp"

namespace gridtools {
    namespace sid {

        enum class property { origin, strides, bounds_validator, ptr_diff, strides_kind, bounds_validator_kind };

        namespace synthetic_impl_ {
            template <property Property, class T>
            struct mixin;

            template <class T>
            struct mixin<property::origin, T> {
                T m_val;
                friend constexpr GT_FUNCTION T sid_get_origin(mixin const &obj) noexcept { return obj.m_val; }
            };

            template <class T>
            struct mixin<property::strides, T> {
                T m_val;
                friend constexpr GT_FUNCTION T sid_get_strides(mixin const &obj) noexcept { return obj.m_val; }
            };

            template <class T>
            struct mixin<property::bounds_validator, T> {
                T m_val;
                friend constexpr GT_FUNCTION T sid_get_bounds_validator(mixin const &obj) noexcept { return obj.m_val; }
            };

            template <class T>
            struct mixin<property::ptr_diff, T> {
                friend T sid_get_ptr_diff(mixin const &) { return {}; }
            };

            template <class T>
            struct mixin<property::strides_kind, T> {};
            template <class T>
            T sid_get_strides_kind(mixin<property::strides_kind, T> const &);

            template <class T>
            struct mixin<property::bounds_validator_kind, T> {};
            template <class T>
            T sid_get_bounds_validator_kind(mixin<property::bounds_validator_kind, T> const &);

            template <property>
            struct unique {};

            template <property Property, class T>
            struct unique_mixin : mixin<Property, T>, unique<Property> {
                unique_mixin() = default;
                unique_mixin(unique_mixin const &) = default;
                unique_mixin(unique_mixin &&) = default;
                unique_mixin &operator=(unique_mixin const &) = default;
                unique_mixin &operator=(unique_mixin &&) = default;

                template <class U>
                constexpr GT_FUNCTION unique_mixin(U &&obj) noexcept
                    : mixin<Property, T>{const_expr::forward<U>(obj)} {}
            };

            template <class...>
            struct synthetic;

            template <>
            struct synthetic<> {
                template <property Property, class T>
                constexpr GT_FUNCTION synthetic<unique_mixin<Property, T>> set() const &&noexcept {
                    return synthetic{};
                }

                template <property Property, class T>
                constexpr GT_FUNCTION synthetic<unique_mixin<Property, decay_t<T>>> set(T &&val) const &&noexcept {
                    return {const_expr::forward<T>(val), synthetic{}};
                }
            };

            template <class Mixin, class... Mixins>
            struct synthetic<Mixin, Mixins...> : Mixin, Mixins... {

                synthetic() = default;

                constexpr GT_FUNCTION synthetic(synthetic<Mixins...> const &&src) noexcept
                    : Mixins(const_expr::move(src))... {}

                template <class T>
                constexpr GT_FUNCTION synthetic(T &&val, synthetic<Mixins...> const &&src) noexcept
                    : Mixin{std::forward<T>(val)}, Mixins(const_expr::move(src))... {}

                template <property Property, class U>
                constexpr GT_FUNCTION synthetic<unique_mixin<Property, U>, Mixin, Mixins...> set() const &&noexcept {
                    return {const_expr::move(*this)};
                }

                template <property Property, class T>
                constexpr GT_FUNCTION synthetic<unique_mixin<Property, decay_t<T>>, Mixin, Mixins...> set(
                    T &&val) const &&noexcept {
                    return {const_expr::forward<T>(val), const_expr::move(*this)};
                }
            };
        } // namespace synthetic_impl_

        /**
         *  A tiny EDSL for creating SIDs from the parts described in the concept.
         *
         *  Usage:
         *
         *  auto my_sid = synthetic()
         *      .set<property::origin>(origin)
         *      .set<property::strides>(strides)
         *      .set<property::bounds_validator>(the_bounds_validator)
         *      .set<property::ptr_diff, ptr_diff>()
         *      .set<property::strides_kind, strides_kind>()
         *      .set<property::bounds_validator_kind, bounds_validator_kind>();
         *
         *  only `set<property::origin>` is required. Other `set`'s can be skipped.
         *  `set`'s can go in any order. `set` of the given property can participate at most once.
         *  Duplicated property `set`'s cause compiler error.
         *  Works both in run time and in compile time.
         */
        constexpr GT_FUNCTION synthetic_impl_::synthetic<> synthetic() { return {}; }
    } // namespace sid
} // namespace gridtools
