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
#include "arg_metafunctions_fwd.hpp"
#include "arg_fwd.hpp"

namespace gridtools {

    /**
     * @struct arg_hods_data_field_h
     * high order metafunction of arg_holds_data_field
     */
    template < typename Arg >
    struct arg_holds_data_field_h {
        typedef typename arg_holds_data_field< typename Arg::type >::type type;
    };

    template < uint_t I, typename Storage, typename Location, bool Temporary >
    struct arg_holds_data_field_h< arg< I, Storage, Location, Temporary > > {
        typedef typename arg_holds_data_field< arg< I, Storage, Location, Temporary > >::type type;
    };

    // metafunction to access the storage type given the arg
    template < typename T >
    struct get_storage_from_arg;

    template < unsigned I, typename T, typename L, bool B >
    struct get_storage_from_arg< arg< I, T, L, B > > {
        typedef T type;
    };

    template < unsigned I, typename T, typename L, bool B >
    struct get_storage_from_arg< arg< I, std::vector< T >, L, B > > {
        typedef T type;
    };

    // metafunction to access the metadata type given the arg
    template < typename T >
    struct get_storage_info_from_arg {
        static_assert(is_arg< T >::value, "Given type is not an arg.");
        typedef typename get_storage_from_arg< T >::type storage_t;
        typedef typename storage_t::storage_info_t type;
    };

} // namespace gridtools
