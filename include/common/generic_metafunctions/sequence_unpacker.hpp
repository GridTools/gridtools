/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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
#include <boost/mpl/at.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/pop_front.hpp>
#include "../defs.hpp"
#include "variadic_typedef.hpp"

namespace gridtools {
#ifdef CXX11_ENABLED
    /*
     * converts a mpl sequence of types into a variadic_typedef of a variadic pack of types
     * Example sequence_unpacker< int,float >::type == variadic_typedef< int, float >
     */
    template < typename Seq, typename... Args >
    struct sequence_unpacker {
        GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< Seq >::value > 0 || sizeof...(Args) > 0), GT_INTERNAL_ERROR);

        template < typename Seq_ >
        struct rec_unpack {
            typedef typename sequence_unpacker< typename boost::mpl::pop_front< Seq_ >::type,
                Args...,
                typename boost::mpl::at_c< Seq_, 0 >::type >::type type;
        };

        template < typename... Args_ >
        struct get_variadic_args {
            using type = variadic_typedef< Args... >;
        };

        typedef typename boost::mpl::eval_if_c< (boost::mpl::size< Seq >::value > 0),
            rec_unpack< Seq >,
            get_variadic_args< Args... > >::type type;
    };
#endif

} // namespace gridtools
