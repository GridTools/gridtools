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

#include "../../common/generic_metafunctions/unzip.hpp"
#include "../../common/gt_assert.hpp"

namespace gridtools {

    template <typename Layout, uint_t... Dims>
    struct meta_storage_cache {

        typedef storage_info_interface<0, Layout> meta_storage_t;
        typedef Layout layout_t;
        GRIDTOOLS_STATIC_ASSERT(layout_t::masked_length == sizeof...(Dims),
            GT_INTERNAL_ERROR_MSG("Mismatch in layout length and passed number of dimensions."));

      public:
        GT_FUNCTION
        constexpr meta_storage_cache() {}

        GT_FUNCTION
        static constexpr uint_t padded_total_length() { return meta_storage_t(Dims...).padded_total_length(); }

        template <ushort_t Id>
        GT_FUNCTION static constexpr int_t stride() {
            return meta_storage_t(Dims...).template stride<Id>();
        }

        template <typename... D, typename std::enable_if<is_all_integral<D...>::value, int>::type = 0>
        GT_FUNCTION constexpr int_t index(D... args_) const {
            return meta_storage_t(Dims...).index(args_...);
        }

        template <ushort_t Id>
        GT_FUNCTION static constexpr int_t dim() {
            return meta_storage_t(Dims...).template total_length<Id>();
        }
    };
} // namespace gridtools
