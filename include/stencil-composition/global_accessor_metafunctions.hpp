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

#include "global_accessor_fwd.hpp"
#include "accessor_metafunctions.hpp"
#include "global_accessor.hpp"

namespace gridtools {
    template < typename Type >
    struct is_global_accessor : boost::false_type {};

    template < uint_t I, enumtype::intend Intend >
    struct is_global_accessor< global_accessor< I, Intend > > : boost::true_type {};

    template < typename Global, typename... Args >
    struct is_global_accessor< global_accessor_with_arguments< Global, Args... > > : boost::true_type {};

    template < typename T >
    struct is_global_accessor_with_arguments : boost::false_type {};

    template < typename Global, typename... Args >
    struct is_global_accessor_with_arguments< global_accessor_with_arguments< Global, Args... > > : boost::true_type {};

    template < typename T >
    struct is_accessor_impl< T, typename std::enable_if< is_global_accessor< T >::value >::type > : boost::mpl::true_ {
    };

    template < ushort_t ID, enumtype::intend Intend, typename ArgsMap >
    struct remap_accessor_type< global_accessor< ID, Intend >, ArgsMap > {
        //        typedef global_accessor< ID, Intend > accessor_t;
        typedef global_accessor< _impl::get_remap_accessor_id< ID, ArgsMap >(), Intend > type;
    };

    template < typename GlobalAcc, typename ArgsMap, typename... Args >
    struct remap_accessor_type< global_accessor_with_arguments< GlobalAcc, Args... >, ArgsMap > {
        //        typedef global_accessor_with_arguments< GlobalAcc, Args... > accessor_t;
        typedef global_accessor_with_arguments<
            global_accessor< _impl::get_remap_accessor_id< GlobalAcc::index_t::value, ArgsMap >(), GlobalAcc::intent >,
            Args... > type;
    };

} // namespace gridtools
