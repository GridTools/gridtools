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

#include "../../common/defs.hpp"
#include "../../common/gt_math.hpp"
#include "../../common/host_device.hpp"
#include "../../common/selector.hpp"

namespace gridtools {
    namespace impl {

        /**
         * @brief Computes a unique identifier (to be used for metastorages) given a list of index values
         */
        template <uint_t Pos>
        GT_FUNCTION constexpr long long compute_uuid_selector(int cnt) {
            return 0;
        }

        /**
         * @brief Computes a unique identifier (to be used for metastorages) given a list of index values
         */
        template <uint_t Pos, typename... Int>
        GT_FUNCTION constexpr long long compute_uuid_selector(int cnt, int val0, Int... val) {
            return (cnt == 4)
                       ? 0
                       : ((val0 == 1)
                                 ? gt_pow<Pos>::apply((long long)2) + compute_uuid_selector<Pos + 1>(cnt + 1, val...)
                                 : compute_uuid_selector<Pos + 1>(cnt + 1, val...));
        }

        /**
         * Computes a unique identifier (to be used for metastorages) given the location type and a dim selector
         * that determines the dimensions of the layout map which are activated.
         * Only the first 4 dimension of the selector are considered, since the iteration space of the backend
         * does not make use of indices beyond the space dimensions
         */
        template <int_t LocationTypeIndex, typename Selector>
        struct compute_uuid {};

        template <int_t LocationTypeIndex, bool... B>
        struct compute_uuid<LocationTypeIndex, selector<B...>> {
            static constexpr ushort_t value =
                enumtype::metastorage_library_indices_limit + LocationTypeIndex + compute_uuid_selector<2>(0, B...);
        };
    } // namespace impl
} // namespace gridtools
