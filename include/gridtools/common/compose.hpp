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
#ifndef GT_COMMON_COMPOSE_HPP_
#define GT_COMMON_COMPOSE_HPP_

#include "defs.hpp"
#include "generic_metafunctions/utility.hpp"
#include "host_device.hpp"

#define GT_FILENAME <gridtools/common/compose.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif // GT_COMMON_COMPOSE_HPP_
#else  // GT_TARGET_ITERATING

namespace gridtools {
    GT_TARGET_NAMESPACE {
        namespace compose_impl_ {
            template <class... Funs>
            struct composed_f;

            template <class F, class G>
            struct composed_f<F, G> {
                F m_f;
                G m_g;

                template <class... Args>
                GT_CONSTEXPR GT_TARGET GT_FORCE_INLINE decltype(auto) operator()(Args &&... args) const {
                    return m_f(m_g(wstd::forward<Args>(args)...));
                }
            };

            template <class F, class... Fs>
            struct composed_f<F, Fs...> : composed_f<F, composed_f<Fs...>> {
                GT_CONSTEXPR GT_TARGET GT_FORCE_INLINE composed_f(F f, Fs... fs)
                    : composed_f<F, composed_f<Fs...>>{wstd::move(f), {wstd::move(fs)...}} {}
            };
        } // namespace compose_impl_

        /// Make function composition from provided functions
        ///
        /// compose(a, b, c)(x, y) <==> a(b(c(x, y)))
        ///
        template <class... Funs>
        GT_CONSTEXPR GT_TARGET GT_FORCE_INLINE compose_impl_::composed_f<Funs...> compose(Funs && ... funs) {
            return {wstd::forward<Funs>(funs)...};
        }

        template <class Fun>
        GT_CONSTEXPR GT_TARGET GT_FORCE_INLINE Fun compose(Fun && fun) {
            return fun;
        }
    }
} // namespace gridtools

#endif // GT_TARGET_ITERATING
