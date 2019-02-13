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
                constexpr GT_TARGET GT_FORCE_INLINE auto operator()(Args &&... args) const
                    GT_AUTO_RETURN(m_f(m_g(const_expr::forward<Args>(args)...)));
            };

            template <class F, class... Fs>
            struct composed_f<F, Fs...> : composed_f<F, composed_f<Fs...>> {
                constexpr GT_TARGET GT_FORCE_INLINE composed_f(F f, Fs... fs)
                    : composed_f<F, composed_f<Fs...>>{const_expr::move(f), {const_expr::move(fs)...}} {}
            };
        } // namespace compose_impl_

        /// Make function composition from provided functions
        ///
        /// compose(a, b, c)(x, y) <==> a(b(c(x, y)))
        ///
        template <class... Funs>
        constexpr GT_TARGET GT_FORCE_INLINE compose_impl_::composed_f<Funs...> compose(Funs && ... funs) {
            return {const_expr::forward<Funs>(funs)...};
        }

        template <class Fun>
        constexpr GT_TARGET GT_FORCE_INLINE Fun compose(Fun && fun) {
            return fun;
        }
    }
} // namespace gridtools

#endif // GT_TARGET_ITERATING
