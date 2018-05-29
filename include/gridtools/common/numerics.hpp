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
#ifndef _NUMERICS_H_
#define _NUMERICS_H_

#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {
    namespace _impl {
        /** \ingroup common
            @{
            \defgroup numerics Compile-Time Numerics
            @{
        */

        /** @brief Compute 3^I at compile time
            \tparam I Exponent
        */
        template < uint_t I >
        struct static_pow3;

        template <>
        struct static_pow3< 0 > {
            static const int value = 1;
        };

        template <>
        struct static_pow3< 1 > {
            static const int value = 3;
        };

        template < uint_t I >
        struct static_pow3 {
            static const int value = 3 * static_pow3< I - 1 >::value;
        };

        /** @brief provide a constexpr version of std::ceil
            \param num Float numner to ceil
         */
        GT_FUNCTION constexpr int static_ceil(float num) {
            return (static_cast< float >(static_cast< int >(num)) == num)
                       ? static_cast< int >(num)
                       : static_cast< int >(num) + ((num > 0) ? 1 : 0);
        }
        /** @} */
        /** @} */
    }
}

#endif
