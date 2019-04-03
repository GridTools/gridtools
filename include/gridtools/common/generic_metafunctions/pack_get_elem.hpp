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

#include <type_traits>

#include "../defs.hpp"
#include "../host_device.hpp"

namespace gridtools {
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
    template <std::size_t Idx>
    struct pack_get_elem {
        template <class First, class... Others>
        GT_FORCE_INLINE static constexpr auto apply(First &&, Others &&... others)
            GT_AUTO_RETURN(pack_get_elem<Idx - 1>::apply(std::forward<Others>(others)...));
    };

    template <>
    struct pack_get_elem<0> {
        template <class First, class... Others>
        GT_FORCE_INLINE static constexpr auto apply(First &&first, Others &&...)
            GT_AUTO_RETURN(std::forward<First>(first));
    };
    /** @} */
    /** @} */
    /** @} */
} // namespace gridtools
