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
#include "../../common/host_device.hpp"
#include "../../common/generic_metafunctions/is_all_integrals.hpp"
#include <vector>

namespace gridtools {
    namespace impl_ {
        /*
         * Recursively building a position within the iteration space
         */
        template < size_t CurDim >
        struct recursive_iterate {
            template < typename Functor, size_t NTotal, typename... Position >
            static GT_FUNCTION void apply(Functor f, const gridtools::array< uint_t, NTotal > &sizes, Position... pos) {
                GRIDTOOLS_STATIC_ASSERT((is_all_integral< Position... >::value), GT_INTERNAL_ERROR);
                for (uint_t i = 0; i < sizes[CurDim - 1]; ++i) {
                    recursive_iterate< CurDim - 1 >::apply(f, sizes, i, pos...);
                }
            }
        };

        // recursion termination
        template <>
        struct recursive_iterate< 0 > {
            GT_NV_EXEC_CHECK_DISABLE
            template < typename Functor, size_t NTotal, typename... Position >
            static GT_FUNCTION void apply(Functor f, const gridtools::array< uint_t, NTotal > &sizes, Position... pos) {
                // position is fully build -> call functor
                f(pos...);
            }

            // case where iteration space is empty (0-dimensional array is passed)
            template < typename Functor, size_t NTotal, typename... Position >
            static GT_FUNCTION void apply(Functor f, const gridtools::array< uint_t, NTotal > &sizes) {
                // noop
            }
        };
    }

    template < typename Functor, size_t Size >
    GT_FUNCTION void iterate(const gridtools::array< uint_t, Size > &sizes, Functor f) {
        impl_::recursive_iterate< Size >::apply(f, sizes);
    }
}
