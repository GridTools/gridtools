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
namespace gridtools {

    /**@brief Expression computing the integral exponent of the first arguments
       for this expression the second argument is an integer (this might, and probably will, be relaxed if needed)
    */
    template < typename ArgType1, int Exponent >
    struct expr_pow : public unary_expr< ArgType1 > {
        typedef unary_expr< ArgType1 > super;
        GT_FUNCTION
        constexpr expr_pow(ArgType1 const &first_operand) : super(first_operand) {}
        static const int exponent = Exponent;

        template < typename Arg1 >
        GT_FUNCTION constexpr expr_pow(expr_pow< Arg1, Exponent > const &other)
            : super(other) {}

#ifndef __CUDACC__
      private:
#endif
        GT_FUNCTION
        constexpr expr_pow() {}
#ifndef __CUDACC__
        static char constexpr op[] = "^2 ";
        typedef string_c< print, op > operation;

      public:
        // currying and recursion (this gets inherited)
        using to_string = concatenate< tokens::open_par, ArgType1, tokens::closed_par, operation >;
#endif
    };

    template < typename ArgType1, int Exponent >
    struct is_unary_expr< expr_pow< ArgType1, Exponent > > : boost::mpl::true_ {};

    namespace expressions {

        /** power expression*/
        template < int exponent,
            typename ArgType1,
            typename boost::disable_if< typename boost::is_arithmetic< ArgType1 >::type, int >::type = 0 >
        GT_FUNCTION constexpr expr_pow< ArgType1, exponent > pow(ArgType1 arg1) {
            return expr_pow< ArgType1, exponent >(arg1);
        }

        namespace evaluation {

            /**
               @brief partial specializations for integer
               Here we do not use the typedef int_t, because otherwise the interface would be polluted with casting
               (the user would have to cast all the literal types (-1, 0, 1, 2 .... ) to int_t before using them in the
               expression)
            */

            template < typename IterateDomain,
                typename ArgType1 /*typename IntType, IntType*/
                ,
                int exponent /*, typename boost::enable_if<typename boost::is_integral<IntType>::type, int >::type=0 */ >
            GT_FUNCTION auto static constexpr value(IterateDomain &it_domain, expr_pow< ArgType1, exponent > const &arg)
                -> decltype(gt_pow< exponent >::apply(it_domain(arg.first_operand))) {
                return gt_pow< exponent >::apply(it_domain(arg.first_operand));
            }

            // automatic differentiation
            /** power derivative evaluation*/
            template < typename IterateDomain, typename ArgType1, int Exponent >
            GT_FUNCTION auto static constexpr value(
                IterateDomain &it_domain, expr_derivative< expr_pow< ArgType1, Exponent > > const &arg)
                -> decltype(it_domain(expr_pow< ArgType1, Exponent - 1 >(arg.first_operand) *
                                      expr_derivative< ArgType1 >(arg.first_operand))) {
                return (it_domain(expr_pow< ArgType1, Exponent - 1 >(arg.first_operand) *
                                  expr_derivative< ArgType1 >(arg.first_operand)));
            }

        } // namespace evaluation
    }     // namespace expressions

} // namespace gridtools
