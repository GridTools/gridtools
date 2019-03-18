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

#include <array>

#include <boost/mpl/accumulate.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/vector.hpp>

#include "../../common/generic_metafunctions/repeat_template.hpp"
#include "../../common/gt_assert.hpp"
#include "../../common/variadic_pack_metafunctions.hpp"

namespace gridtools {
    /** \ingroup storage
     * @{
     */

    /**
     *  @brief A class that is used to pass halo information to the storage info.
     *  E.g., Lets say we want to retrieve a storage_info instance with a halo
     *  of 2 in I and J direction. We have to pass following type to the storage_facility
     *  halo<2,2,0>. The I and J dimensions of the storage will be extended by 2
     *  in + and - direction.
     *  @tparam N variadic list of halo sizes
     */
    template <uint_t... N>
    struct halo {

        /**
         * @brief member function used to query the halo size of a given dimension
         * @tparam V Dimension or coordinate to query
         * @return halo size
         */
        template <uint_t V>
        GT_FUNCTION static constexpr uint_t at() {
            GT_STATIC_ASSERT(
                (V < sizeof...(N)), GT_INTERNAL_ERROR_MSG("Out of bounds access in halo type discovered."));
            return get_value_from_pack(V, N...);
        }

        /**
         * @brief member function used to query the halo size of a given dimension
         * @param V Dimension or coordinate to query
         * @return halo size
         */
        GT_FUNCTION static constexpr uint_t at(uint_t V) { return get_value_from_pack(V, N...); }

        /**
         * @brief member function used to query the number of dimensions. E.g., a halo
         * type with 3 entries cannot be passed to a <3 or >3 dimensional storage_info.
         * @return number of dimensions
         */
        GT_FUNCTION static constexpr uint_t size() { return sizeof...(N); }
    };

    /**
     *  @brief Used to generate a zero initialzed halo. Used as a default value for storage info halo.
     */
    template <uint_t Cnt>
    using zero_halo = typename repeat_template_c<0, Cnt, halo>::type;

    /* used to check if a given type is a halo type */
    template <typename T>
    struct is_halo : boost::mpl::false_ {};

    template <uint_t... N>
    struct is_halo<halo<N...>> : boost::mpl::true_ {};

    /**
     * @}
     */
} // namespace gridtools
