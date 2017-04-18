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

/**
   @file
   @brief On all boundary the values are copied from the last data field to the first. Minimum 2 fields.
*/

namespace gridtools {

    struct copy_boundary {

        /**   @brief On all boundary the values are copied from the last data field to the first. Minimum 2 fields. */
        template < typename Direction, typename DataField0, typename DataField1 >
        GT_FUNCTION void operator()(
            Direction, DataField0 &data_field0, DataField1 const &data_field1, uint_t i, uint_t j, uint_t k) const {
            data_field0(i, j, k) = data_field1(i, j, k);
        }

        template < typename Direction, typename DataField0, typename DataField1, typename DataField2 >
        GT_FUNCTION void operator()(Direction,
            DataField0 &data_field0,
            DataField1 &data_field1,
            DataField2 const &data_field2,
            uint_t i,
            uint_t j,
            uint_t k) const {
            data_field0(i, j, k) = data_field2(i, j, k);
            data_field1(i, j, k) = data_field2(i, j, k);
        }
    };

} // namespace gridtools
