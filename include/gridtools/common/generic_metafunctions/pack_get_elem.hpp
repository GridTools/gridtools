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
