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
#ifndef GT_COMMON_GENERIC_METAFUNCTIONS_FOR_EACH_HPP_
#define GT_COMMON_GENERIC_METAFUNCTIONS_FOR_EACH_HPP_

#include "host_device.hpp"

#define GT_FILENAME <gridtools/common/for_each.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif
#else

namespace gridtools {
    GT_TARGET_NAMESPACE {
        namespace for_each_impl_ {
            template <class>
            struct for_each_f;

#if __cplusplus < 201703
            template <template <class...> class L>
            struct for_each_f<L<>> {
                template <class F>
                GT_TARGET GT_FORCE_INLINE GT_TARGET_CONSTEXPR void operator()(F &&) const {}
            };

            template <template <class...> class L, class... Ts>
            struct for_each_f<L<Ts...>> {
                template <class F>
                GT_TARGET GT_FORCE_INLINE GT_TARGET_CONSTEXPR void operator()(F &&f) const {
                    using array_t = int[sizeof...(Ts)];
                    (void)array_t{((void)f(Ts()), 0)...};
                }
            };
#else
            template <template <class...> class L, class... Ts>
            struct for_each_f<L<Ts...>> {
                template <class F>
                GT_TARGET GT_FORCE_INLINE GT_TARGET_CONSTEXPR void operator()(F &&f) const {
                    (..., f(Ts()));
                }
            };
#endif
        } // namespace for_each_impl_

        template <class L>
        constexpr for_each_impl_::for_each_f<L> for_each = {};
    }
} // namespace gridtools

#endif
