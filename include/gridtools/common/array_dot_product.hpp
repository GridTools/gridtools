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
#include "./array.hpp"
#include "./generic_metafunctions/gt_integer_sequence.hpp"
#include "./generic_metafunctions/accumulate.hpp"

namespace gridtools {
    namespace _impl {
        template <typename T, typename U, size_t D, size_t ...Is>
        GT_FUNCTION constexpr auto dot_impl(array< T, D > const& a, array< U, D > const& b, gt_integer_sequence<size_t, Is...>) -> decltype(accumulate(plus_functor{}, (a[Is]*b[Is])...)) {
            return accumulate(plus_functor{}, (a[Is]*b[Is])...);
        }
    }

    /**
     * @brief dot product for gridtools::array (enabled for all arithmetic types)
     *
     * \param First array
     * \param Second Array
     *
     * \return Value corresponding to the first array value type
     */
    template < typename T, typename U, size_t D, typename std::enable_if< std::is_arithmetic< T >::value and std::is_arithmetic< U >::value, T >::type = 0 >
    GT_FUNCTION constexpr T array_dot_product(array< T, D > const& a, array< U, D > const &b)
    {
        return _impl::dot_impl( a, b, typename make_gt_integer_sequence<size_t, D>::type{} );
    }
} // namespace gridtools
