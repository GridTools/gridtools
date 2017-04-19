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

namespace gridtools {

#ifdef CXX11_ENABLED
    /**@file
    @brief implementation of a compile time string.

    It consists of a sequence of static const char* and of a "callable" operator, which can be i.e. calling printf, or
    doing
    compile-time operations.
    */

    template < char const *S >
    struct static_string {
        static const constexpr char *value = S;
    };

    /**@brief this struct allows to perform operations on static char arrays (e.g. print them as concatenated strings)
     */
    template < typename Callable, typename... Known >
    struct string {

        GT_FUNCTION
        constexpr string() {}

        // operator calls the constructor of the arg_type
        GT_FUNCTION
        static void apply() { Callable::apply(Known::value...); }
    };

    /**@brief this struct allows to perform operations on static char arrays (e.g. print them as concatenated strings)
     */
    template < typename Callable, char const *... Known >
    struct string_c {

        GT_FUNCTION
        constexpr string_c() {}

        // static constexpr char* m_known[]={Known...};
        // operator calls the constructor of the arg_type
        GT_FUNCTION
        static void apply() { Callable::apply(Known...); }
    };

    /** apply the given operator to all strings recursively*/
    template < typename First, typename... Strings >
    struct concatenate {

        GT_FUNCTION
        static void apply() {
            First::to_string::apply();
            concatenate< Strings... >::apply();
        }
    };

    /** apply the given operator to all strings recursively*/
    template < typename String >
    struct concatenate< String > {

        GT_FUNCTION
        static void apply() { String::to_string::apply(); }
    };

    /**@brief struct to recursively print all the strings contained in the gridtools::string template arguments*/
    struct print {

        static void apply() {}

        template < typename... S >
        GT_FUNCTION static void apply(const char *first, S... s) {
            printf("%s", first);
            apply(s...);
        }

        template < typename IntType,
            typename... S,
            typename boost::enable_if< typename boost::is_integral< IntType >::type, int >::type = 0 >
        GT_FUNCTION static void apply(IntType first, S... s) {
            printf("%d", first);
            apply(s...);
        }

        template < typename FloatType,
            typename... S,
            typename boost::enable_if< typename boost::is_floating_point< FloatType >::type, int >::type = 0 >
        GT_FUNCTION static void apply(FloatType first, S... s) {
            printf("%f", first);
            apply(s...);
        }
    };
#endif // CXX11_ENABLED

    /**@brief simple function that copies a string **/
    inline char const *malloc_and_copy(char const *src) {
        char *dst = new char[strlen(src) + 1];
        strcpy(dst, src);
        return dst;
    }
} // namespace gridtools
