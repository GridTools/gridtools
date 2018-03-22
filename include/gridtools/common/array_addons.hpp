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
#include <vector>
#include <array>

namespace gridtools {
    template < typename T, size_t D >
    std::ostream &operator<<(std::ostream &s, array< T, D > const &a) {
        s << " {  ";
        for (int i = 0; i < D - 1; ++i) {
            s << a[i] << ", ";
        }
        s << a[D - 1] << "  } ";

        return s;
    }

    template < typename T, size_t D >
    std::vector< T > to_vector(array< T, D > const &a) {
        std::vector< T > v(D);
        for (int i = 0; i < D; ++i) {
            v.at(i) = a[i];
        }
        return v;
    }

    namespace impl {
        template < typename Value >
        struct array_initializer {
            template < int Idx >
            struct type {
                type() = delete;

                template < long unsigned int ndims >
                constexpr static Value apply(const std::array< Value, ndims > data) {
                    return data[Idx];
                }
            };
        };
    }
} // namespace gridtools

template < typename T, typename U, size_t D >
bool same_elements(gridtools::array< T, D > const &a, gridtools::array< U, D > const &b) {
    // shortcut
    if (a.size() != b.size())
        return false;

    // sort and check for equivalence
    gridtools::array< T, D > a0 = a;
    gridtools::array< U, D > b0 = b;
    std::sort(a0.begin(), a0.end());
    std::sort(b0.begin(), b0.end());
    return std::equal(a0.begin(), a0.end(), b0.begin());
}
