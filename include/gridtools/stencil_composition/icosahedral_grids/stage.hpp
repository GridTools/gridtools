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
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../meta.hpp"
#include "../accessor_intent.hpp"
#include "../arg.hpp"
#include "../extent.hpp"
#include "../has_apply.hpp"
#include "../location_type.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/multi_shift.hpp"
#include "dim.hpp"
#include "icosahedral_topology.hpp"
#include "on_neighbors.hpp"

/**
 *   @file
 *
 *   Stage concept represents elementary functor from the backend implementor point of view.
 *   Stage concept for icosahedral grid is defined similar as for structured grid (with some additions)
 *
 *   Stage must have the nested `extent_t` type or an alias that has to model Extent concept.
 *   The meaning: the stage should be computed in the area that is extended from the user provided computation area by
 *   that much.
 *
 *   Stage also have static `exec` method that accepts an object by reference that models IteratorDomain.
 *   `exec` should execute an elementary functor for all colors from the grid point that IteratorDomain points to.
 *   precondition: IteratorDomain should point to the first color.
 *   postcondition: IteratorDomain still points to the first color.
 *
 *   Stage has templated variation of `exec` which accept color number as a first template parameter. This variation
 *   does not iterate on colors; it executes an elementary functor for the given color.
 *   precondition: IteratorDomain should point to the same color as one in exec parameter.
 *
 *   Stage has netsted metafunction contains_color<Color> that evaluates to std::false_type if for the given color
 *   the elementary function is not executed.
 *
 *   Note that the Stage is (and should stay) backend independent. The core of gridtools passes stages [split by k-loop
 *   intervals and independent groups] to the backend in the form of compile time only parameters.
 *
 *   TODO(anstaf): add `is_stage<T>` trait
 */

namespace gridtools {

    namespace stage_impl_ {
        template <class Ptr, class Strides, class Keys, class LocationType, int_t Color>
        struct evaluator {
            Ptr const &m_ptr;
            Strides const &m_strides;

            template <class Key, class Offset>
            GT_FUNCTION decltype(auto) get_ref(Offset offset) const {
                auto ptr = host_device::at_key<Key>(m_ptr);
                sid::multi_shift<Key>(ptr, m_strides, wstd::move(offset));
                return *ptr;
            }

            template <class Accessor>
            GT_FUNCTION decltype(auto) operator()(Accessor const &acc) const {
                return apply_intent<Accessor::intent_v>(get_ref<meta::at_c<Keys, Accessor::index_t::value>>(acc));
            }

            template <class Accessor, class Offset>
            GT_FUNCTION decltype(auto) neighbor(Offset const &offset) const {
                return apply_intent<Accessor::intent_v>(get_ref<meta::at_c<Keys, Accessor::index_t::value>>(offset));
            }

            template <class ValueType, class LocationTypeT, class Reduction, class... Accessors>
            GT_FUNCTION ValueType operator()(
                on_neighbors<ValueType, LocationTypeT, Reduction, Accessors...> onneighbors) const {
                static constexpr auto offsets = connectivity<LocationType, LocationTypeT, Color>::offsets();
                for (auto &&offset : offsets)
                    onneighbors.m_value = onneighbors.m_function(neighbor<Accessors>(offset)..., onneighbors.m_value);
                return onneighbors.m_value;
            }

            static constexpr int_t color = Color;
        };

        template <class Functor, class PlhMap>
        struct stage {
            GT_STATIC_ASSERT(has_apply<Functor>::value, GT_INTERNAL_ERROR);
            using location_t = typename Functor::location;
            using num_colors_t = typename location_t::n_colors;

            template <class Ptr, class Strides>
            GT_FUNCTION void operator()(Ptr ptr, Strides const &strides) const {
                host_device::for_each<meta::make_indices<num_colors_t>>([&](auto color) {
                    using eval_t = evaluator<Ptr, Strides, PlhMap, location_t, decltype(color)::value>;
                    Functor::apply(eval_t{ptr, strides});
                    sid::shift(ptr, sid::get_stride<dim::c>(strides), integral_constant<int_t, 1>());
                });
            }
        };
    } // namespace stage_impl_
    using stage_impl_::stage;
} // namespace gridtools
