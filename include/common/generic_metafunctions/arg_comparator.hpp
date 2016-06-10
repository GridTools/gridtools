/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
#include "../../gridtools.hpp"
namespace gridtools {

    /**
       @brief struct containing conditionals for several types.

       To be used with e.g. mpl::sort
    */
    struct arg_comparator {
        template < typename T1, typename T2 >
        struct apply;

        /**specialization for storage pairs*/
        template < typename T1, typename T2, typename T3, typename T4 >
        struct apply< arg_storage_pair< T1, T2 >, arg_storage_pair< T3, T4 > >
            : public boost::mpl::bool_< (T1::index_type::value < T3::index_type::value) > {};

        /**specialization for storage placeholders*/
        template < ushort_t I1, typename T1, ushort_t I2, typename T2 >
        struct apply< arg< I1, T1 >, arg< I2, T2 > > : public boost::mpl::bool_< (I1 < I2) > {};

        /**specialization for static integers*/
        template < typename T, T T1, T T2 >
        struct apply< boost::mpl::integral_c< T, T1 >, boost::mpl::integral_c< T, T2 > >
            : public boost::mpl::bool_< (T1 < T2) > {};
    };
}
