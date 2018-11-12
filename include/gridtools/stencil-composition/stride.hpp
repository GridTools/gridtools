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

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/type_traits.hpp"
#include "../common/host_device.hpp"

namespace gridtools {

    /**
     * This function computes the stride along the the given axis/coordinate.
     *
     * @tparam StorageInfo The storage info to be used.
     * @tparam Coordinate The axis along which the stride should be computed.
     * @tparam StridesCached Type of the strides array.
     *
     * @param strides Array of stride values.
     *
     * @return The stride along the given axis: 0 if the axis is masked, 1 if the axis is the contiguous one, run
     * time value read from `strides` otherwise.
     */
    template <typename StorageInfo, int_t Coordinate, typename StridesCached>
    GT_FUNCTION constexpr int_t stride(StridesCached const &RESTRICT strides) {
        using layout_t = typename StorageInfo::layout_t;

        /* get the maximum integer value in the layout map */
        using layout_max_t = std::integral_constant<int, layout_t::max()>;

        /* get the layout map value at the given coordinate */
        using layout_val_t = std::integral_constant<int, layout_t::template at_unsafe<Coordinate>()>;

        /* check if we are at a masked-out value (-> stride == 0) */
        using is_masked_t = bool_constant<layout_val_t::value == -1>;
        /* check if we are at the maximum value (-> stride == 1) */
        using is_max_t = bool_constant<layout_max_t::value == layout_val_t::value>;

        /* return constants for masked and max coordinates, otherwise lookup stride */
        return is_masked_t::value ? 0 : is_max_t::value ? 1 : strides[layout_val_t::value];
    }

} // namespace gridtools
