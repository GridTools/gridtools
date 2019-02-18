/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "../../common/defs.hpp"
#include "expr_base.hpp"

namespace gridtools {
    namespace expressions {

        /** \addtogroup stencil-composition
            @{
            \addtogroup expressions
            @{
        */
        struct plus_f {
            template <class Lhs, class Rhs>
            GT_FUNCTION constexpr auto operator()(Lhs const &lhs, Rhs const &rhs) const GT_AUTO_RETURN(lhs + rhs);
            template <class Arg>
            GT_FUNCTION constexpr auto operator()(Arg const &arg) const GT_AUTO_RETURN(+arg);
        };

        template <class Lhs, class Rhs>
        GT_FUNCTION constexpr auto operator+(Lhs lhs, Rhs rhs) GT_AUTO_RETURN(make_expr(plus_f{}, lhs, rhs));

        template <class Arg>
        GT_FUNCTION constexpr auto operator+(Arg arg) GT_AUTO_RETURN(make_expr(plus_f{}, arg));
        /** @} */
        /** @} */
    } // namespace expressions
} // namespace gridtools
