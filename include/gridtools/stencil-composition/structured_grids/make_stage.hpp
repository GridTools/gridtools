/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <tuple>

#ifdef GT_PEDANTIC
#include <boost/mpl/size.hpp>
#endif

#include "../../common/defs.hpp"
#include "../../meta/type_traits.hpp"
#include "../arg.hpp"
#include "./esf.hpp"

namespace gridtools {

    /**
     * @brief Function to create a descriptor for a stage (ij-pass over a grid)
     *
     * Extents are derived from the stage definitions.
     */
    template <typename ESF, typename... Args>
    constexpr esf_descriptor<ESF, std::tuple<Args...>> make_stage(Args...) {
        GT_STATIC_ASSERT(conjunction<is_plh<Args>...>::value, "Malformed make_stage");
#ifdef PEDANTIC // find a way to enable this check also with generic accessors
        GT_STATIC_ASSERT(sizeof...(Args) == boost::mpl::size<typename ESF::arg_list>::value,
            "wrong number of arguments passed to the make_esf");
#endif
        return {};
    }

    /**
     * @brief Function to create a descriptor for a stage (ij-pass over a grid)
     *
     * Extents are given as a template argument.
     */
    template <typename ESF, typename Extent, typename... Args>
    constexpr esf_descriptor_with_extent<ESF, Extent, std::tuple<Args...>> make_stage_with_extent(Args...) {
        GT_STATIC_ASSERT(conjunction<is_plh<Args>...>::value, "Malformed make_stage");
#ifdef PEDANTIC // find a way to enable this check also with generic accessors
        GT_STATIC_ASSERT(sizeof...(Args) == boost::mpl::size<typename ESF::arg_list>::value,
            "wrong number of arguments passed to the make_esf");
#endif
        return {};
    }
} // namespace gridtools
