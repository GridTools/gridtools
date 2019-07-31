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

#include <tuple>
#include <type_traits>

#include "../common/defs.hpp"
#include "arg.hpp"
#include "esf.hpp"
#include "extent.hpp"

namespace gridtools {
    /**
     * @brief Function to create a descriptor for a stage (ij-pass over a grid)
     *
     * Extent is given as a template argument.
     * If Extent is not provided it is derived from the stage definitions.
     */
    template <class Esf, class Extent = void, class... Args>
    constexpr std::tuple<esf_descriptor<Esf, std::tuple<Args...>, Extent>> make_stage(Args...) {
        GT_STATIC_ASSERT(conjunction<is_plh<Args>...>::value, "Malformed make_stage");
        GT_STATIC_ASSERT(sizeof...(Args) == meta::length<typename Esf::param_list>::value,
            "wrong number of arguments passed to the make_stage");
        GT_STATIC_ASSERT(std::is_void<Extent>::value || is_extent<Extent>::value, "Invalid Extent type");
        return {};
    }

    /**
     * @brief Function to create a descriptor for a stage (ij-pass over a grid)
     *
     * Extents are given as a template argument.
     */
    template <typename Esf, typename Extent, typename... Args>
    constexpr auto make_stage_with_extent(Args... args) {
        return make_stage<Esf, Extent>(args...);
    }
} // namespace gridtools
