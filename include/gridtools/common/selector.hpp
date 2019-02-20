/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <boost/mpl/bool.hpp>

#include "defs.hpp"

namespace gridtools {

    /** \ingroup common
        @{
        \ingroup layout
        @{
    */

    /**
     *  @brief A class that is used as a selector when selecting which dimensions should be masked.
     *  E.g., Lets say we want to have a 3-dimensional storage but second dimension should be masked
     *  we have to pass following selector selector<1,0,1> to the storage-facility.
     *  @tparam Bitmask bitmask defining the masked and unmasked dimensions
     */
    template <bool... Bitmask>
    struct selector {
        static constexpr uint_t size = sizeof...(Bitmask);
    };

    template <typename T>
    struct is_selector : boost::mpl::false_ {};

    template <bool... Bitmask>
    struct is_selector<selector<Bitmask...>> : boost::mpl::true_ {};
    /** @} */
    /** @} */
} // namespace gridtools
