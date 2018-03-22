/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include <type_traits>

#include "../../common/generic_metafunctions/meta.hpp"
#include "../accessor_fwd.hpp"
#include "../global_accessor_fwd.hpp"

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
    template < class Op, class... Args >
    struct expr;

    template < class Op, class Arg >
    struct expr< Op, Arg > {
        Arg m_arg;
    };

    template < class Op, class Lhs, class Rhs >
    struct expr< Op, Lhs, Rhs > {
        Lhs m_lhs;
        Rhs m_rhs;
    };

    namespace expressions {

        template < class Arg >
        using expr_or_accessor = std::integral_constant< bool,
            meta::is_instantiation_of< expr >::apply< Arg >::value || is_accessor< Arg >::value ||
                is_global_accessor< Arg >::value || is_global_accessor_with_arguments< Arg >::value >;

        template < class Op,
            class... Args,
            typename std::enable_if< meta::disjunction< expr_or_accessor< Args >... >::value, int >::type = 0 >
        GT_FUNCTION constexpr expr< Op, Args... > make_expr(Op, Args... args) {
            return {args...};
        }

        namespace evaluation {
            template < class Eval,
                class Arg,
                typename std::enable_if< std::is_arithmetic< Arg >::value, int >::type = 0 >
            GT_FUNCTION constexpr Arg apply_eval(Eval &, Arg arg) {
                return arg;
            }

            template < class Eval,
                class Arg,
                typename std::enable_if< !std::is_arithmetic< Arg >::value, int >::type = 0 >
            GT_FUNCTION constexpr auto apply_eval(Eval &eval, Arg const &arg) GT_AUTO_RETURN(eval(arg));

            template < class Eval, class Op, class Arg >
            GT_FUNCTION constexpr auto value(Eval &eval, expr< Op, Arg > const &arg)
                GT_AUTO_RETURN(Op{}(eval(arg.m_arg)));

            template < class Eval, class Op, class Lhs, class Rhs >
            GT_FUNCTION constexpr auto value(Eval &eval, expr< Op, Lhs, Rhs > const &arg)
                GT_AUTO_RETURN(Op{}(apply_eval(eval, arg.m_lhs), apply_eval(eval, arg.m_rhs)));
        }
    }
    /** @} */
    /** @} */
}
