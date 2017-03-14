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

    /**@brief Expression enabling the direct access to the storage.

       The offsets only (without the index) identify the memory address to be used
    */
    template < typename ArgType1 >
    struct expr_direct_access : public unary_expr< ArgType1 > {
        typedef unary_expr< ArgType1 > super;
        GT_FUNCTION
        constexpr expr_direct_access(ArgType1 const &first_operand) : super(first_operand) {}

        template < typename Arg1 >
        GT_FUNCTION constexpr expr_direct_access(expr_direct_access< Arg1 > const &other)
            : super(other) {}

#ifndef __CUDACC__
      private:
#endif
        GT_FUNCTION
        constexpr expr_direct_access() {}
#ifndef __CUDACC__
        static char constexpr op[] = " !";
        typedef string_c< print, op > operation;

      public:
        // currying and recursion (this gets inherited)
        using to_string = concatenate< operation, tokens::open_par, ArgType1, tokens::closed_par >;
#endif
    };

    template < typename ArgType1 >
    struct is_unary_expr< expr_direct_access< ArgType1 > > : boost::mpl::true_ {};

    namespace expressions {

        /** direct access expression*/
        template < typename ArgType1,
            typename boost::disable_if< no_expr_nor_accessor_types< ArgType1, int >, int >::type = 0 >
        GT_FUNCTION constexpr expr_direct_access< ArgType1 > operator!(ArgType1 arg1) {
            return expr_direct_access< ArgType1 >(arg1);
        }

    } // namespace expressions

} // namespace gridtools
