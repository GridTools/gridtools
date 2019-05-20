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

#include "../../meta.hpp"
#include "../arg.hpp"
#include "../esf_fwd.hpp"
#include "extent.hpp"

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
    } // namespace esf_impl_

    /**
     * @brief Descriptors for Elementary Stencil Function (ESF)
     */
    template <class EsfFunction, class Args>
    struct esf_descriptor {
        GT_STATIC_ASSERT((meta::all_of<is_plh, Args>::value),
            "wrong types for the list of parameter placeholders check the make_stage syntax");
        GT_STATIC_ASSERT(esf_impl_::has_param_list<EsfFunction>::type::value,
            "The type param_list was not found in a user functor definition. All user functors must have a type alias "
            "called \'param_list\', which is an MPL vector containing the list of accessors defined in the functor "
            "(NOTE: the \'global_accessor\' types are excluded from this list). Example: \n\n using v1=accessor<0>; \n "
            "using v2=global_accessor<1>; \n using v3=accessor<2>; \n using "
            "param_list=make_param_list<v1, v3>;");
        GT_STATIC_ASSERT(esf_impl_::check_param_list<typename EsfFunction::param_list>::value,
            "The list of accessors in a user functor (i.e. the param_list type to be defined on each functor) does not "
            "have increasing index");
        GT_STATIC_ASSERT(meta::length<typename EsfFunction::param_list>::value == meta::length<Args>::value,
            "The number of actual aerguments should match the number of parameters.");

        using esf_function_t = EsfFunction;
        using args_t = Args;
    };

    template <class EsfFunction, class Args>
    struct is_esf_descriptor<esf_descriptor<EsfFunction, Args>> : std::true_type {};
} // namespace gridtools
