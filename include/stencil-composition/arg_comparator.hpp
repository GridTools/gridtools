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
        template < ushort_t I1, typename T1, typename L1, ushort_t I2, typename T2, typename L2 >
        struct apply< arg< I1, T1, L1 >, arg< I2, T2, L2 > > : public boost::mpl::bool_< (I1 < I2) > {};

        /**specialization for static integers*/
        template < typename T, T T1, T T2 >
        struct apply< boost::mpl::integral_c< T, T1 >, boost::mpl::integral_c< T, T2 > >
            : public boost::mpl::bool_< (T1 < T2) > {};
    };
}
