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

#include "../../common/gt_assert.hpp"
#include "../../common/variadic_pack_metafunctions.hpp"
#include "../../meta/repeat.hpp"

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

    namespace _impl {
        template <class>
        struct list_to_halo;
        template <template <class...> class L, uint_t... Is>
        struct list_to_halo<L<std::integral_constant<uint_t, Is>...>> {
            using type = halo<Is...>;
        };
    } // namespace _impl

    /**
     *  @brief Used to generate a zero initialzed halo. Used as a default value for storage info halo.
     */
    template <uint_t Cnt>
    using zero_halo =
        typename _impl::list_to_halo<GT_META_CALL(meta::repeat_c, (Cnt, std::integral_constant<uint_t, 0>))>::type;

    /* used to check if a given type is a halo type */
    template <typename T>
    struct is_halo : std::false_type {};

    template <uint_t... N>
    struct is_halo<halo<N...>> : std::true_type {};

    /**
     * @}
     */
} // namespace gridtools
