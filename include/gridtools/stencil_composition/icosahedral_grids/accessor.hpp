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
    template <uint_t ID, intent Intent, typename LocationType, typename Extent = extent<>, uint_t FieldDimensions = 4>
    struct accessor : accessor_base<FieldDimensions> {
        GT_STATIC_ASSERT(is_location_type<LocationType>::value, GT_INTERNAL_ERROR);
        using index_t = static_uint<ID>;
        static constexpr intent intent_v = Intent;
        using extent_t = Extent;
        using location_type = LocationType;
        static constexpr uint_t value = ID;

        /**inheriting all constructors from accessor_base*/
        using accessor_base<FieldDimensions>::accessor_base;
    };

    template <uint_t ID, intent Intent, typename LocationType, typename Extent, uint_t FieldDimensions>
    GT_META_CALL(meta::repeat_c, (FieldDimensions, int_t))
    tuple_to_types(accessor<ID, Intent, LocationType, Extent, FieldDimensions> const &);

    template <uint_t ID, intent Intent, typename LocationType, typename Extent, uint_t FieldDimensions>
    meta::always<accessor<ID, Intent, LocationType, Extent, FieldDimensions>> tuple_from_types(
        accessor<ID, Intent, LocationType, Extent, FieldDimensions> const &);

    template <uint_t ID, typename LocationType, typename Extent = extent<>, uint_t FieldDimensions = 4>
    using in_accessor = accessor<ID, intent::in, LocationType, Extent, FieldDimensions>;

    template <uint_t ID, typename LocationType, uint_t FieldDimensions = 4>
    using inout_accessor = accessor<ID, intent::inout, LocationType, extent<>, FieldDimensions>;

    template <uint_t ID, intent Intent, typename LocationType, typename Extent, uint_t FieldDimensions>
    struct is_accessor<accessor<ID, Intent, LocationType, Extent, FieldDimensions>> : std::true_type {};
} // namespace gridtools
