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

    /**@brief Expression retrieving the maximum over a specific dimension*/
    template < typename ArgType1 >
    struct expr_derivative : public unary_expr< ArgType1 > {
        typedef unary_expr< ArgType1 > super;
        GT_FUNCTION
        constexpr expr_derivative(ArgType1 const &first_operand) : super(first_operand) {}

        template < typename Arg1 >
        GT_FUNCTION constexpr expr_derivative(expr_derivative< Arg1 > const &other)
            : super(other) {}

#ifndef __CUDACC__
      private:
#endif
        GT_FUNCTION
        constexpr expr_derivative(){};
#ifndef __CUDACC__
        static char constexpr op[] = " D";
        typedef string_c< print, op > operation;

      public:
        // currying and recursion (this gets inherited)
        using to_string = concatenate< operation, tokens::open_par, ArgType1, tokens::closed_par >;
#endif
    };

    template < typename ArgType1 >
    struct is_unary_expr< expr_derivative< ArgType1 > > : boost::mpl::true_ {};

    namespace expressions {

        template < typename ArgType1,
            typename boost::disable_if<
                boost::mpl::not_< boost::mpl::or_< is_accessor< ArgType1 >, is_expr< ArgType1 > > >,
                int >::type = 0 >
        GT_FUNCTION constexpr expr_derivative< ArgType1 > D(ArgType1 arg1) {
            return expr_derivative< ArgType1 >(arg1);
        }

        namespace evaluation {

            /** derivative evaluation*/
            template < typename IterateDomain, typename ArgType1 >
            GT_FUNCTION static float_type constexpr value(IterateDomain const &it_domain,
                expr_derivative< ArgType1 > const &arg,
                typename boost::enable_if< typename is_accessor< ArgType1 >::type, int >::type = 0) {
                return 1.;
                // each operator implements its own derivative
            }

        } // namespace evaluation
    }     // namespace expressions
} // namespace gridtools
