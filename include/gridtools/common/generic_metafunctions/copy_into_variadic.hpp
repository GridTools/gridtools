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

#include <boost/mpl/copy.hpp>
#include <boost/mpl/inserter.hpp>

#include "../defs.hpp"

namespace gridtools {

    namespace _impl {
        struct variadic_push_back {
            template < class, class >
            struct apply;
            template < template < class... > class L, class... Ts, class T >
            struct apply< L< Ts... >, T > {
                using type = L< Ts..., T >;
            };
        };
    }

    /// Helper to copy MPL sequence to a variadic typelist
    template < class Src, class Dst >
    struct lazy_copy_into_variadic : boost::mpl::copy< Src, boost::mpl::inserter< Dst, _impl::variadic_push_back > > {};

    template < class Src, class Dst >
    using copy_into_variadic =
        typename boost::mpl::copy< Src, boost::mpl::inserter< Dst, _impl::variadic_push_back > >::type;
}
