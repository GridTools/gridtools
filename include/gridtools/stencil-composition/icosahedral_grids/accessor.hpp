/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../meta/always.hpp"
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

        friend meta::always<accessor> tuple_from_types(accessor const &) { return {}; }
    };

    template <uint_t ID, typename LocationType, typename Extent = extent<>, ushort_t FieldDimensions = 4>
    using in_accessor = accessor<ID, intent::in, LocationType, Extent, FieldDimensions>;

    template <uint_t ID, typename LocationType, ushort_t FieldDimensions = 4>
    using inout_accessor = accessor<ID, intent::inout, LocationType, extent<>, FieldDimensions>;

    template <uint_t ID, intent Intent, typename LocationType, typename Extent, ushort_t FieldDimensions>
    struct is_accessor<accessor<ID, Intent, LocationType, Extent, FieldDimensions>> : std::true_type {};
} // namespace gridtools
