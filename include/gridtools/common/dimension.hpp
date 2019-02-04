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

#include <type_traits>

#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup dimension Dimension
        @{
    */

    /**
       @brief The following struct defines one specific component of a field
       It contains a direction (compile time constant, specifying the ID of the component),
       and a value (runtime value, which is storing the offset in the given direction).
    */
    template <ushort_t Coordinate>
    struct dimension {
        GRIDTOOLS_STATIC_ASSERT(Coordinate != 0, "The coordinate values passed to the accessor start from 1");

        GT_FUNCTION constexpr dimension() : value(0) {}

        template <typename IntType>
        GT_FUNCTION constexpr dimension(IntType val) : value{(int_t)val} {}

        dimension(dimension const &) = default;

        static constexpr ushort_t index = Coordinate;
        int_t value;
    };

    template <typename T>
    struct is_dimension : std::false_type {};

    template <ushort_t Id>
    struct is_dimension<dimension<Id>> : std::true_type {};

    /** @} */
    /** @} */
} // namespace gridtools
