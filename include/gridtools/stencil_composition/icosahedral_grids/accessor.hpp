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
    template <uint_t Id, intent Intent, class LocationType, class Extent = extent<>, uint_t FieldDimensions = 4>
    class accessor : public accessor_base<Id, Intent, Extent, FieldDimensions> {
        using base_t = typename accessor::accessor_base;

      public:
        GT_STATIC_ASSERT(is_location_type<LocationType>::value, GT_INTERNAL_ERROR);
        using location_t = LocationType;

        GT_DECLARE_DEFAULT_EMPTY_CTOR(accessor);
        accessor(accessor const &) = default;
        accessor(accessor &&) = default;

        GT_FUNCTION constexpr accessor(array<int_t, FieldDimensions> src) : base_t(std::move(src)) {}

        template <uint_t J, uint_t... Js>
        GT_FUNCTION constexpr accessor(dimension<J> src, dimension<Js>... srcs) : base_t(src, srcs...) {}
    };

    template <uint_t ID, intent Intent, typename LocationType, typename Extent, uint_t FieldDimensions>
    meta::repeat_c<FieldDimensions, int_t> tuple_to_types(
        accessor<ID, Intent, LocationType, Extent, FieldDimensions> const &);

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
