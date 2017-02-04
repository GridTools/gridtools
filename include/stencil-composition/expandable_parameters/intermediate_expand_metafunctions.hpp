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

/** @file metafunctions used in @ref gridtools::intermediate_expand*/

namespace gridtools {
    namespace _impl {

        // ********* metafunctions ************
        template < typename T >
        struct is_expandable_parameters : boost::mpl::false_ {};

        template < typename BaseStorage, ushort_t N >
        struct is_expandable_parameters< expandable_parameters< BaseStorage, N > > : boost::mpl::true_ {};

        template < typename BaseStorage >
        struct is_expandable_parameters< std::vector< pointer< BaseStorage > > > : boost::mpl::true_ {};

        template < typename T >
        struct is_expandable_arg : boost::mpl::false_ {};

        template < uint_t N, typename Storage, typename Condition >
        struct is_expandable_arg< arg< N, Storage, Condition > > : is_expandable_parameters< Storage > {};

        template < uint_t N, typename Storage >
        struct is_expandable_arg< arg< N, Storage > > : is_expandable_parameters< Storage > {};

        template < typename T >
        struct get_basic_storage {
            typedef typename arg2storage<T>::type::basic_type type;
        };

        template < typename T >
        struct get_storage {
            typedef typename T::storage_type type;
        };

        template < typename T >
        struct get_index {
            typedef typename T::index_type type;
            static const uint_t value = T::index_type::value;
        };

        template < enumtype::platform B >
        struct create_arg;

        template <>
        struct create_arg< enumtype::Host > {
            template < typename T, typename ExpandFactor >
            struct apply {
                typedef arg< get_index< T >::value,
                    storage< expandable_parameters< typename get_basic_storage< T >::type, ExpandFactor::value > > >
                    type;
            };

            template < typename T, typename ExpandFactor, uint_t ID >
            struct apply< arg< ID, std::vector< pointer< no_storage_type_yet< T > > > >, ExpandFactor > {
                typedef arg< ID,
                    no_storage_type_yet< storage<
                        expandable_parameters< typename T::basic_type, ExpandFactor::value > > > > type;
            };
        };

        template <>
        struct create_arg< enumtype::Cuda > {
            template < typename T, typename ExpandFactor >
            struct apply {
                typedef arg< get_index< T >::value,
                    storage< expandable_parameters< typename get_basic_storage< T >::type, ExpandFactor::value > > >
                    type;
            };

            template < uint_t ID, typename T, typename ExpandFactor >
            struct apply< arg< ID, std::vector< pointer< no_storage_type_yet< T > > > >, ExpandFactor > {
                typedef arg< ID,
                    no_storage_type_yet< storage<
                        expandable_parameters< typename T::basic_type, ExpandFactor::value > > > > type;
            };
        };

    } // namespace _impl
} // namespace gridtools
