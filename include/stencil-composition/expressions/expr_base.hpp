/*
GridTools Libraries

Copyright (c) 2016, GridTools Consortium
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
namespace gridtools {

    struct tokens {
        static char constexpr par_o[] = "(";
        static char constexpr par_c[] = ")";
        using open_par = string_c< print, par_o >;
        using closed_par = string_c< print, par_c >;
    };
    /** \section expressions Expressions Definition
        @{
        This is the base class of a binary expression, containing the instances of the two arguments.
        The expression should be a static constexpr object, instantiated once for all at the beginning of the run.
    */
    template < typename First, typename Second >
    struct binary_expr {

        /**@brief generic expression constructor*/
        GT_FUNCTION
        constexpr binary_expr(First const &first_, Second const &second_)
            : first_operand(first_), second_operand(second_) {}

        template < typename Arg1, typename Arg2 >
        GT_FUNCTION constexpr binary_expr(binary_expr< Arg1, Arg2 > const &other)
            : first_operand(other.first_operand), second_operand(other.second_operand) {}

        First const first_operand;
        Second const second_operand;

#ifndef __CUDACC__
      private:
#endif
        /**@brief default empty constructor*/
        GT_FUNCTION
        constexpr binary_expr() {}
    };

    template < typename Arg >
    struct is_binary_expr : boost::mpl::false_ {};

    template < typename Arg1, typename Arg2 >
    struct is_binary_expr< binary_expr< Arg1, Arg2 > > : boost::mpl::true_ {};

    template < typename ArgType1 >
    struct unary_expr {

        /**@brief generic expression constructor*/
        GT_FUNCTION
        constexpr unary_expr(ArgType1 const &first_operand) : first_operand{first_operand} {}

        template < typename Arg1 >
        GT_FUNCTION constexpr unary_expr(unary_expr< Arg1 > const &other)
            : first_operand(other.first_operand) {}

        ArgType1 const first_operand;

#ifndef __CUDACC__
      private:
#endif
        /**@brief default empty constructor*/
        GT_FUNCTION
        constexpr unary_expr() {}
    };

    template < typename ArgType1, typename ArgType2, typename ArgType3 >
    struct ternary_expr {

        /**@brief generic expression constructor*/
        GT_FUNCTION
        constexpr ternary_expr(
            ArgType1 const &first_operand, ArgType2 const &second_operand, ArgType3 const &third_operand)
            : first_operand{first_operand}, second_operand{second_operand}, third_operand{third_operand} {}

        template < typename Arg1, typename Arg2, typename Arg3 >
        GT_FUNCTION constexpr ternary_expr(ternary_expr< Arg1, Arg2, Arg3 > const &other)
            : first_operand(other.first_operand), second_operand(other.second_operand),
              third_operand(other.third_operand) {}

        ArgType1 const first_operand;
        ArgType2 const second_operand;
        ArgType3 const third_operand;
#ifndef __CUDACC__
      private:
#endif
        /**@brief default empty constructor*/
        GT_FUNCTION
        constexpr ternary_expr() {}
    };

    template < typename Arg >
    struct is_unary_expr : boost::mpl::bool_< Arg::size == 1 > {};

    template < typename Arg >
    struct is_expr : boost::mpl::false_ {};

    template < typename... Args >
    struct is_expr< binary_expr< Args... > > : boost::mpl::true_ {};

    template < typename Arg >
    struct is_expr< unary_expr< Arg > > : boost::mpl::true_ {};

    template < typename Arg >
    struct is_accessor;

    /**
       @namespace expressions
       @brief Overloaded operators
       The algebraic operators are overloaded in order to deal with expressions. To enable these operators the user has
       to use the namespace expressions.*/
    namespace expressions {

        template < typename Arg1, typename Arg2 >
        using both_arithmetic_types =
            typename boost::mpl::and_< boost::is_arithmetic< Arg1 >, boost::is_arithmetic< Arg2 > >::type;

        template < typename Arg1, typename Arg2 >
        using no_expr_types =
            typename boost::mpl::not_< typename boost::mpl::or_< is_expr< Arg1 >, is_expr< Arg2 > >::type >::type;

        template < typename Arg1, typename Arg2 >
        using no_accessor_types = typename boost::mpl::not_<
            typename boost::mpl::or_< is_accessor< Arg1 >, is_accessor< Arg2 > >::type >::type;

        template < typename Arg1, typename Arg2 >
        using no_expr_nor_accessor_types =
            typename boost::mpl::and_< no_accessor_types< Arg1, Arg2 >, no_expr_types< Arg1, Arg2 > >::type;

    } // namespace expressions

    /**fwd declaration*/
    template < typename Arg >
    struct expr_derivative;

} // namespace gridtools
