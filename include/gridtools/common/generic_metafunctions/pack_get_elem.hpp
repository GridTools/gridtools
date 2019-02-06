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
#include "../defs.hpp"
#include "variadic_typedef.hpp"
#include <boost/mpl/if.hpp>

namespace gridtools {

    namespace impl {

        template <typename ReturnType, uint_t Idx>
        GT_FUNCTION constexpr ReturnType pack_get_elem_(uint_t pos) {
            return ReturnType();
        }

        template <typename ReturnType, uint_t Idx, typename First, typename... ElemTypes>
        GT_FUNCTION constexpr ReturnType pack_get_elem_(uint_t pos, First first, ElemTypes... elems) {
            return (pos == Idx) ? first : pack_get_elem_<ReturnType, Idx>(pos + 1, elems...);
        }

        // This metafunction is only for reporting a readable error message to a call to pack_get_elem with
        // a negative index
        template <int_t Idx>
        struct pack_get_elem_null {
            template <typename... ElemTypes>
            GT_FUNCTION static constexpr int apply(ElemTypes... elems) {
                GT_STATIC_ASSERT((Idx < 0), "Error: trying to retrieve a element of a pack with a negative index");
                return 0;
            }
        };

        template <int_t Idx>
        struct pack_get_elem_elem {
            template <typename... ElemTypes>
            GT_FUNCTION static constexpr typename variadic_typedef<ElemTypes...>::template get_elem<Idx>::type apply(
                ElemTypes... elems) {
                return pack_get_elem_<typename variadic_typedef<ElemTypes...>::template get_elem<Idx>::type, Idx>(
                    0, elems...);
            }
        };
    } // namespace impl

    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \ingroup variadic
        @{
    */
    /*
     * metafunction to retrieve a certain element of a variadic pack
     * Example of use:
     * pack_get_elem<2>::apply(3,4,7) == 7
     */
    template <int_t Idx>
    struct pack_get_elem
        : boost::mpl::if_c<(Idx < 0), impl::pack_get_elem_null<Idx>, impl::pack_get_elem_elem<Idx>>::type {};
    /** @} */
    /** @} */
    /** @} */
} // namespace gridtools
