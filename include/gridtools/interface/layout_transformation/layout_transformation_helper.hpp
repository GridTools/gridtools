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
#include "../../common/defs.hpp"
#include <vector>

namespace gridtools {
    namespace impl {
        /**
         * @brief copy std::vector to (potentially bigger) gridtools::array
         */
        template <size_t MaxDim>
        gridtools::array<gridtools::uint_t, MaxDim> vector_to_array(
            const std::vector<uint_t> &v, gridtools::uint_t init_value) {
            assert(MaxDim >= v.size() && "array too small");

            gridtools::array<gridtools::uint_t, MaxDim> a;
            std::fill(a.begin(), a.end(), init_value);
            std::copy(v.begin(), v.end(), a.begin());
            return a;
        }

        template <size_t MaxDim>
        gridtools::array<gridtools::uint_t, MaxDim> vector_to_dims_array(const std::vector<uint_t> &v) {
            return vector_to_array<MaxDim>(v, 1);
        }

        template <size_t MaxDim>
        gridtools::array<gridtools::uint_t, MaxDim> vector_to_strides_array(const std::vector<uint_t> &v) {
            return vector_to_array<MaxDim>(v, 0);
        }
    } // namespace impl
} // namespace gridtools
