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

#include "array.hpp"

namespace gridtools {
    /** \ingroup common
        @{
     */

    namespace impl_ {
        template < typename ForceType, typename... Types >
        struct forced_or_common_type {
            using type = ForceType;
        };

        template < typename... Types >
        struct forced_or_common_type< void, Types... > {
            using type = typename std::common_type< Types... >::type;
        };
    }

    /** \ingroup array
        @{
    */

    /** \brief Facility to make an array given a variadic list of
        values.

        Facility to make an array given a variadic list of
        values.An explicit template argument can be used to force the
        value type of the array. The list of values passed to the
        function must have a common type or be covertible to the
        explici value type if that is specified. The size of the array
        is the length of the list of values.

        \tparam ForceType Value type of the resulting array (optional)
        \param values List of values to put in the array. The length of the list set the size of the array.
     */
    template < typename ForceType = void, typename... Types >
    constexpr GT_FUNCTION
        gridtools::array< typename impl_::forced_or_common_type< ForceType, Types... >::type, sizeof...(Types) >
            make_array(Types... values) {
        return gridtools::array< typename impl_::forced_or_common_type< ForceType, Types... >::type, sizeof...(Types) >{
            static_cast< typename impl_::forced_or_common_type< ForceType, Types... >::type >(values)...};
    }

    /** @} */
    /** @} */
}
