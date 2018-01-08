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

#include "../../common/array.hpp"
#include "../../common/generic_metafunctions/gt_integer_sequence.hpp"
#include "../../common/defs.hpp"
#include <vector>

namespace gridtools {
    namespace impl_ {
        template < typename F, typename Array, std::size_t... I >
        GT_FUNCTION void apply_array_to_variadic_impl(F f, const Array &a, gt_integer_sequence< std::size_t, I... >) {
            f(a[I]...);
        }
        template < typename F,
            typename Array,
            typename Indices = typename make_gt_integer_sequence< std::size_t, Array::size() >::type >
        GT_FUNCTION void apply_array_to_variadic(F f, const Array &a) {
            apply_array_to_variadic_impl(f, a, Indices{});
        }

        template < size_t Size >
        GT_FUNCTION bool is_empty_set(const gridtools::array< uint_t, Size > &sizes) {
            uint_t size = 1;
            for (size_t i = 0; i < Size; ++i)
                size *= sizes[i];
            return size == 0 || Size == 0;
        }
    }

    /**
     * @brief Iterate Functor over the range {0...sizes[0]-1, 0...sizes[1]-1,...}
     */
    template < typename Functor, size_t Size >
    GT_FUNCTION void iterate(const gridtools::array< uint_t, Size > &sizes, Functor f) {
        gridtools::array< uint_t, Size > index = {};
        size_t index_index = 0;

        if (!impl_::is_empty_set(sizes)) {
            impl_::apply_array_to_variadic(f, index);
            while (true) {
                if (index[index_index] + 1 < sizes[index_index]) {
                    index[index_index]++;
                    while (index_index > 0) {
                        index_index--;
                        index[index_index] = 0;
                    }
                    impl_::apply_array_to_variadic(f, index);
                } else {
                    if (index_index + 1 < Size) {
                        index_index++;
                    } else {
                        break;
                    }
                }
            }
        }
    }
}
