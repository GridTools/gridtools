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

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../meta/type_traits.hpp"
#include "../is_accessor.hpp"

namespace gridtools {

    /** \ingroup stencil-composition
        @{
        \ingroup expressions
        @{
    */

    /** Expressions Definition
        This is the base class of a binary expression, containing the instances of the two arguments.
        The expression should be a static constexpr object, instantiated once for all at the beginning of the run.
    */
    template <class Op, class... Args>
    struct expr;

    template <class Op, class Arg>
    struct expr<Op, Arg> {
        Arg m_arg;
    };

    template <class Op, class Lhs, class Rhs>
    struct expr<Op, Lhs, Rhs> {
        Lhs m_lhs;
        Rhs m_rhs;
    };

    namespace expressions {

        template <class>
        struct is_expr : std::false_type {};
        template <class... Ts>
        struct is_expr<expr<Ts...>> : std::true_type {};

        template <class Arg>
        using expr_or_accessor = bool_constant<is_expr<Arg>::value || is_accessor<Arg>::value>;

        template <class Op, class... Args, enable_if_t<disjunction<expr_or_accessor<Args>...>::value, int> = 0>
        GT_FUNCTION expr<Op, Args...> make_expr(Op, Args... args) {
            return {args...};
        }

        namespace evaluation {
            template <class Eval, class Arg, enable_if_t<std::is_arithmetic<Arg>::value, int> = 0>
            GT_FUNCTION Arg apply_eval(Eval &, Arg arg) {
                return arg;
            }

            template <class Eval, class Arg, enable_if_t<!std::is_arithmetic<Arg>::value, int> = 0>
            GT_FUNCTION auto apply_eval(Eval &eval, Arg const &arg) GT_AUTO_RETURN(eval(arg));

            template <class Eval, class Op, class Arg>
            GT_FUNCTION auto value(Eval &eval, expr<Op, Arg> const &arg) GT_AUTO_RETURN(Op{}(eval(arg.m_arg)));

            template <class Eval, class Op, class Lhs, class Rhs>
            GT_FUNCTION auto value(Eval &eval, expr<Op, Lhs, Rhs> const &arg)
                GT_AUTO_RETURN(Op{}(apply_eval(eval, arg.m_lhs), apply_eval(eval, arg.m_rhs)));
        } // namespace evaluation
    }     // namespace expressions
    /** @} */
    /** @} */
} // namespace gridtools
