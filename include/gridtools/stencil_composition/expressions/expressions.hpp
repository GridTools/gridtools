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

#include "../../common/defs.hpp"
#include "../../common/dimension.hpp"

/**@file
   @brief Expression templates definition.
   The expression templates are a method to parse at compile time the mathematical expression given
   by the user, recognizing the structure and building a syntax tree by recursively nesting
   templates.
*/

#include "./expr_divide.hpp"
#include "./expr_minus.hpp"
#include "./expr_plus.hpp"
#include "./expr_pow.hpp"
#include "./expr_times.hpp"

namespace gridtools {

    /** \ingroup stencil-composition
        @{
    */

    /** \defgroup expressions Expressions
        @{
    */

    /** Namespace containing all the compomnents to enable using expressions in stencil operators
     */
    namespace expressions {

        /** Expressions defining the interface for specifiyng a given offset for a specified dimension in plus-direction
           \tparam Coordinate: direction in which to apply the offset
           \param d1 Coordinate id
           \param offset: the offset to be applied in the Coordinate direction
        */
        template <uint_t Coordinate>
        GT_FUNCTION GT_CONSTEXPR dimension<Coordinate> operator+(dimension<Coordinate>, int offset) {
            return {offset};
        }

        /** Expressions defining the interface for specifiyng a given offset for a specified dimension in
           minus-direction
           \tparam Coordinate: direction in which to apply the offset
           \param d1 Coordinate id
           \param offset: the offset to be applied in the Coordinate direction
        */
        template <uint_t Coordinate>
        GT_FUNCTION GT_CONSTEXPR dimension<Coordinate> operator-(dimension<Coordinate>, int offset) {
            return {-offset};
        }
    } // namespace expressions

    /** @} */
    /** @} */

} // namespace gridtools
