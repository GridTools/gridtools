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

#include "../../common/generic_metafunctions/is_sequence_of.hpp"
#include "../../meta.hpp"
#include "../esf_aux.hpp"
#include "../esf_fwd.hpp"
#include "extent.hpp"

/**
   @file
   @brief Descriptors for Elementary Stencil Function (ESF)
*/
namespace gridtools {
    namespace esf_impl_ {
        template <class Param>
        GT_META_DEFINE_ALIAS(param_index, std::integral_constant, (size_t, Param::index_t::value));

        template <class ParamList,
            class Actual = GT_META_CALL(
                meta::rename, (meta::list, GT_META_CALL(meta::transform, (param_index, ParamList)))),
            class Expected = GT_META_CALL(meta::make_indices_for, ParamList)>
        GT_META_DEFINE_ALIAS(check_param_list, std::is_same, (Actual, Expected));

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
        GT_STATIC_ASSERT((is_sequence_of<Args, is_plh>::value),
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

        /** Type member with the mapping between placeholder types (as key) to extents in the operator */
        using args_with_extents_t =
            typename impl::make_arg_with_extent_map<args_t, typename EsfFunction::param_list>::type;
    };

    template <typename ESF, typename ArgArray>
    struct is_esf_descriptor<esf_descriptor<ESF, ArgArray>> : std::true_type {};

    template <typename ESF, typename Extent, typename ArgArray>
    struct esf_descriptor_with_extent : esf_descriptor<ESF, ArgArray> {
        GT_STATIC_ASSERT((is_extent<Extent>::value), "stage descriptor is expecting a extent type");
    };

    template <typename ESF, typename Extent, typename ArgArray>
    struct is_esf_descriptor<esf_descriptor_with_extent<ESF, Extent, ArgArray>> : std::true_type {};

    template <typename ESF>
    struct is_esf_with_extent : std::false_type {
        GT_STATIC_ASSERT(is_esf_descriptor<ESF>::type::value,
            GT_INTERNAL_ERROR_MSG("is_esf_with_extents expects an esf_descripto as template argument"));
    };

    template <typename ESF, typename Extent, typename ArgArray>
    struct is_esf_with_extent<esf_descriptor_with_extent<ESF, Extent, ArgArray>> : std::true_type {};

} // namespace gridtools
