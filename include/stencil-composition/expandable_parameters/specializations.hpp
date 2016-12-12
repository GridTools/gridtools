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

#include "../../gridtools.hpp"

namespace gridtools {

    template < typename Storage, uint_t Size >
    struct expandable_parameters;

    // metafunction to access the storage type given the arg
    template < ushort_t ID, typename T >
    struct arg2storage< arg< ID, std::vector< pointer< T > > > > {
        typedef T type;
    };

    /** metafunction extracting the location type from the storage*/
    template < typename T >
    struct get_location_type< std::vector< T > > {
        typedef typename T::value_type::storage_info_type::index_type type;
    };

#ifdef CXX11_ENABLED
    template < typename Sequence, typename Arg >
    struct insert_if_not_present< Sequence, std::vector< pointer< Arg > > > : insert_if_not_present< Sequence, Arg > {
        using insert_if_not_present< Sequence, Arg >::insert_if_not_present;
    };

    /**
       specialization for expandable parameters
     */
    template < typename T >
    struct storage_holds_data_field< std::vector< pointer< T > > > : boost::mpl::true_ {};

    template < typename T >
    struct is_actual_storage< pointer< std::vector< pointer< T > > > > : public boost::mpl::bool_< !T::is_temporary > {
    };

    template < typename Storage >
    struct is_storage< std::vector< pointer< Storage > > > : is_storage< Storage > {};

    template < typename T >
    struct is_temporary_storage< std::vector< pointer< T > > > : public is_temporary_storage< T > {};

    template < typename T >
    struct is_any_storage< std::vector< T > > : is_any_storage< T > {};

    template < typename T >
    struct get_space_dimensions< std::vector< pointer< T > > > : get_space_dimensions< T > {};
    template < typename T >
    struct get_base_storage< std::vector< pointer< T > > > {
        typedef T type;
    };

#endif

#ifdef CXX11_ENABLED
    template < typename T, uint_t ID >
    struct is_actual_storage< pointer< storage< expandable_parameters< T, ID > > > >
        : public boost::mpl::bool_< !T::is_temporary > {};

    template < typename T, ushort_t Dim >
    struct is_temporary_storage< storage< expandable_parameters< T, Dim > > >
        : public boost::mpl::bool_< T::is_temporary > {};
#endif

} // namespace gridtools
