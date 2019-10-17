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

#include "../meta.hpp"
#include "arg.hpp"
#include "extent.hpp"
#include "location_type.hpp"

/**
   @file
   @brief Descriptors for Elementary Stencil Function (ESF)
*/
namespace gridtools {
    namespace esf_impl_ {
        template <class Param>
        using param_index = std::integral_constant<size_t, Param::index_t::value>;

        template <class ParamList,
            class Actual = meta::rename<meta::list, meta::transform<param_index, ParamList>>,
            class Expected = meta::make_indices_for<ParamList>>
        using check_param_list = std::is_same<Actual, Expected>;

        template <class, class = void>
        struct has_param_list : std::false_type {};

        template <class T>
        struct has_param_list<T, void_t<typename T::param_list>> : std::true_type {};

        template <class Arg, class Accessor>
        using is_same_location = std::is_same<typename Arg::location_t, typename Accessor::location_t>;

        template <class Args, class Accessors>
        using are_same_locations = meta::all<meta::transform<is_same_location, Args, Accessors>>;
    } // namespace esf_impl_

    /**
     * @brief Descriptors for Elementary Stencil Function (ESF)
     */
    template <class EsfFunction, class Args, class Extent = void>
    struct esf_descriptor {
        static_assert(meta::all_of<is_plh, Args>::value,
            "wrong types for the list of parameter placeholders check the make_stage syntax");
        static_assert(esf_impl_::has_param_list<EsfFunction>::type::value,
            "The type param_list was not found in a user functor definition. All user functors must have a type alias "
            "called \'param_list\', which is an MPL vector containing the list of accessors defined in the functor "
            "Example: \n\n using v1=in_accessor<0>; \n\n using v2=inout_accessor<2>; \n using "
            "param_list=make_param_list<v1, v2>;");
        static_assert(esf_impl_::check_param_list<typename EsfFunction::param_list>::value,
            "The list of accessors in a user functor (i.e. the param_list type to be defined on each functor) does not "
            "have increasing index");
        static_assert(meta::length<typename EsfFunction::param_list>::value == meta::length<Args>::value,
            "The number of actual arguments should match the number of parameters.");
        static_assert(esf_impl_::are_same_locations<Args, typename EsfFunction::param_list>::value,
            "Location types of placeholders and accessors must match");

        using esf_function_t = EsfFunction;
        using args_t = Args;
        using extent_t = Extent;
    };

    template <class T>
    using is_esf_descriptor = meta::is_instantiation_of<esf_descriptor, T>;
} // namespace gridtools
