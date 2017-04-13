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
#include <boost/mpl/vector.hpp>
#include <boost/mpl/push_back.hpp>

namespace gridtools {

#ifdef CXX11_ENABLED
    /**
     * @struct variadic_to_vector
     * metafunction that returns a mpl vector from a pack of variadic arguments
     * This is a replacement of using type=boost::mpl::vector<Args ...>, but at the moment nvcc
     * does not properly unpack the last arg of Args... when building the vector. We can eliminate this
     * metafunction once the vector<Args...> works
     */
    template < typename... Args >
    struct variadic_to_vector;

    template < class T, typename... Args >
    struct variadic_to_vector< T, Args... > {
        typedef typename boost::mpl::push_front< typename variadic_to_vector< Args... >::type, T >::type type;
    };

    template < class T >
    struct variadic_to_vector< T > {
        typedef boost::mpl::vector1< T > type;
    };

    template <>
    struct variadic_to_vector<> {
        typedef boost::mpl::vector0<> type;
    };

#endif
}
