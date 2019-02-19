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

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../accessor_base.hpp"
#include "../accessor_intent.hpp"
#include "../extent.hpp"
#include "../is_accessor.hpp"
#include "../location_type.hpp"

namespace gridtools {
    /**
     * This is the type of the accessors accessed by a stencil functor.
     */
    template <uint_t ID, intent Intent, typename LocationType, typename Extent = extent<>, ushort_t FieldDimensions = 4>
    struct accessor : accessor_base<FieldDimensions> {
        GT_STATIC_ASSERT((is_location_type<LocationType>::value), "Error: wrong type");
        using index_t = static_uint<ID>;
        static constexpr intent intent_v = Intent;
        using extent_t = Extent;
        using location_type = LocationType;
        static constexpr uint_t value = ID;
        location_type location() const { return location_type(); }

        /**inheriting all constructors from accessor_base*/
        using accessor_base<FieldDimensions>::accessor_base;

        template <uint_t OtherID, typename std::enable_if<ID != OtherID, int>::type = 0>
        GT_FUNCTION accessor(accessor<OtherID, Intent, LocationType, Extent, FieldDimensions> const &src)
            : accessor_base<FieldDimensions>(src) {}
    };

    template <uint_t ID, typename LocationType, typename Extent = extent<>, ushort_t FieldDimensions = 4>
    using in_accessor = accessor<ID, intent::in, LocationType, Extent, FieldDimensions>;

    template <uint_t ID, typename LocationType, ushort_t FieldDimensions = 4>
    using inout_accessor = accessor<ID, intent::inout, LocationType, extent<>, FieldDimensions>;

    template <uint_t ID, intent Intent, typename LocationType, typename Extent, ushort_t FieldDimensions>
    struct is_accessor<accessor<ID, Intent, LocationType, Extent, FieldDimensions>> : std::true_type {};
} // namespace gridtools
