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
#include "../../common/generic_metafunctions/utility.hpp"
#include "expr_base.hpp"

namespace gridtools {
    namespace expressions {

        /** \addtogroup stencil-composition
            @{
            \addtogroup expressions
            @{
        */
        struct minus_f {
            template <class Lhs, class Rhs>
            GT_FUNCTION constexpr auto operator()(Lhs lhs, Rhs rhs) const GT_AUTO_RETURN(lhs - rhs);
            template <class Arg>
            GT_FUNCTION constexpr auto operator()(Arg arg) const GT_AUTO_RETURN(-arg);
        };

        template <class Lhs, class Rhs>
        GT_FUNCTION constexpr auto operator-(Lhs lhs, Rhs rhs)
            GT_AUTO_RETURN(make_expr(minus_f{}, const_expr::move(lhs), const_expr::move(rhs)));

        template <class Arg>
        GT_FUNCTION constexpr auto operator-(Arg arg) GT_AUTO_RETURN(make_expr(minus_f{}, const_expr::move(arg)));
        /** @} */
        /** @} */
    } // namespace expressions
} // namespace gridtools
