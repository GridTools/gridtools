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
#include "expr_base.hpp"

namespace gridtools {
    namespace expressions {

        /** \ingroup stencil-composition
            @{
            \ingroup expressions
            @{
        */

        /**@brief Evaluator of the division expression*/
        struct divide_f {
            template <class Lhs, class Rhs>
            GT_FUNCTION GT_HOST_CONSTEXPR auto operator()(Lhs const &lhs, Rhs const &rhs) const GT_AUTO_RETURN(lhs / rhs);
        };

        /** Operator overload of the division expression */
        template <class Lhs, class Rhs>
        GT_FUNCTION GT_HOST_CONSTEXPR auto operator/(Lhs lhs, Rhs rhs) GT_AUTO_RETURN(make_expr(divide_f{}, lhs, rhs));
    } // namespace expressions
    /** @} */
    /** @} */
} // namespace gridtools
