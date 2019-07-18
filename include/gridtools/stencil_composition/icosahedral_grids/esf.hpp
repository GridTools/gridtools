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
#include "../../meta.hpp"
#include "../arg.hpp"
#include "../esf_fwd.hpp"
#include "../extent.hpp"
#include "icosahedral_topology.hpp"

namespace gridtools {
    namespace esf_impl_ {
        template <class Arg, class Accessor>
        using is_same_location = std::is_same<typename Arg::location_t, typename Accessor::location_type>;

        template <class Args, class Accessors>
        using are_same_locations = meta::all<meta::transform<is_same_location, Args, Accessors>>;

        template <class, class = void>
        struct has_param_list : std::false_type {};

        template <class T>
        struct has_param_list<T, void_t<typename T::param_list>> : std::true_type {};
    } // namespace esf_impl_

    template <template <uint_t> class EsfFunction, class LocationType, class Args>
    struct esf_descriptor {
        GT_STATIC_ASSERT((meta::all_of<is_plh, Args>::value),
            "wrong types for the list of parameter placeholders check the make_stage syntax");
        GT_STATIC_ASSERT((esf_impl_::are_same_locations<Args, typename EsfFunction<0>::param_list>::value),
            "Location types of placeholders and accessors must match");
        GT_STATIC_ASSERT(esf_impl_::has_param_list<EsfFunction<0>>::type::value,
            "The type param_list was not found in a user functor definition. All user functors must have a type alias "
            "called \'param_list\'.");

        template <uint_t C>
        using esf_function = EsfFunction<C>;

        using location_type = LocationType;
        using args_t = Args;
        using extent_t = void;
    };

    template <template <uint_t> class EsfFunction, class LocationType, class Args>
    struct is_esf_descriptor<esf_descriptor<EsfFunction, LocationType, Args>> : std::true_type {};
} // namespace gridtools
