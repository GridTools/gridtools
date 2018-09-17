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

#include "../common/array.hpp"
#include "../common/host_device.hpp"
#include "../storage/common/data_store_field_metafunctions.hpp"
#include "../storage/data_store_field.hpp"
#include "./accessor_base.hpp"

namespace gridtools {

    /*
     *  This function was located historically in interate_domain_aux.hpp
     *  It was moved first to data_store_field.hpp and after it to the separate file.
     *  The move was done during resolving circular header dependencies.
     *  The original semantics and implementation was not changed (as well as a luck of comments)
     *  The hope is that this function will go away soon (after expandable parameter support will be moved
     *  out of the core of the library).
     */
    template <typename T>
    struct get_datafield_offset {
        template <typename Acc>
        GT_FUNCTION static constexpr uint_t get(Acc const &a) {
            return 0;
        }
    };

    template <typename T, unsigned... N>
    struct get_datafield_offset<data_store_field<T, N...>> {
        template <typename Acc>
        GT_FUNCTION static constexpr uint_t get(Acc const &a) {
            return get_accumulated_data_field_index(gridtools::get<Acc::n_dimensions - 2>(a), N...) +
                   gridtools::get<Acc::n_dimensions - 1>(a);
        }
    };

} // namespace gridtools