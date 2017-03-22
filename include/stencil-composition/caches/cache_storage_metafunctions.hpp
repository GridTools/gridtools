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
#include "../../common/generic_metafunctions/variadic_to_vector.hpp"
#include "../../common/generic_metafunctions/accumulate.hpp"
/**
   @file metafunctions used in the cache_storage class
*/

namespace gridtools {

    namespace _impl {

        template < typename Layout, typename Location, unsigned D1, unsigned D2, unsigned... Rest >
        struct get_meta_storage {
            typedef meta_storage_cache< Layout, D1, Location::n_colors::value, D2, Rest... > type;
        };

        template < typename Layout, unsigned D1, unsigned D2, unsigned... Rest >
        struct get_meta_storage< Layout, enumtype::default_location_type, D1, D2, Rest... > {
            typedef meta_storage_cache< Layout, D1, D2, Rest... > type;
        };

        template < typename Layout, typename Plus, typename Minus, typename Tiles, typename StorageWrapper >
        struct compute_meta_storage;

        /**
           @class computing the correct storage_info type for the cache storage
           \tparam Layout memory layout of the cache storage
           \tparam Plus the positive extents in all directions
           \tparam Plus the negative extents in all directions

           The extents and block size are used to compute the dimension of the cache storage, which is
           all we need.
         */
        template < typename Layout,
            typename P1,
            typename P2,
            typename... Plus,
            typename M1,
            typename M2,
            typename... Minus,
            typename T1,
            typename T2,
            typename... Tiles,
            typename StorageWrapper >
        struct compute_meta_storage< Layout,
            variadic_to_vector< P1, P2, Plus... >,
            variadic_to_vector< M1, M2, Minus... >,
            variadic_to_vector< T1, T2, Tiles... >,
            StorageWrapper > {
            typedef typename StorageWrapper::arg_t::location_t location_t;
            static constexpr unsigned d1 = P1::value - M1::value + T1::value;
            static constexpr unsigned d2 = P2::value - M2::value + T2::value;
            typedef typename get_meta_storage< Layout,
                location_t,
                d1,
                d2,
                ((Plus::value - Minus::value) > 0 ? (Tiles::value - Minus::value + Plus::value) : 1)... >::type type;
        };

        namespace impl {
            template < ushort_t D >
            struct get_layout_map_;

            template <>
            struct get_layout_map_< 2 > {
                typedef layout_map< 1, 0 > type;
            };

            template <>
            struct get_layout_map_< 3 > {
                typedef layout_map< 2, 1, 0 > type;
            };

            template <>
            struct get_layout_map_< 4 > {
                typedef layout_map< 3, 2, 1, 0 > type;
            };

            template <>
            struct get_layout_map_< 5 > {
                typedef layout_map< 4, 3, 2, 1, 0 > type;
            };

            template <>
            struct get_layout_map_< 6 > {
                typedef layout_map< 5, 4, 3, 2, 1, 0 > type;
            };
        }

        template < typename T >
        struct generate_layout_map;

        /**@class automatically generates the layout map for the cache storage. By default
           i and j have the smallest stride. The largest stride is in the field dimension. This reduces bank conflicts.
         */
        template < uint_t... Id >
        struct generate_layout_map< gt_integer_sequence< uint_t, Id... > > {
            typedef layout_map< (sizeof...(Id)-Id - 1)... > type;
        };

    } // namespace _impl
} // namespace gridtools
