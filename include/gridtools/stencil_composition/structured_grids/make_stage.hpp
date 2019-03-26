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
    template <typename Esf, typename... Args>
    constexpr esf_descriptor<Esf, std::tuple<Args...>> make_stage(Args...) {
        GT_STATIC_ASSERT(conjunction<is_plh<Args>...>::value, "Malformed make_stage");
        GT_STATIC_ASSERT(sizeof...(Args) == meta::length<typename Esf::param_list>::value,
            "wrong number of arguments passed to the make_stage");
        return {};
    }

    /**
     * @brief Function to create a descriptor for a stage (ij-pass over a grid)
     *
     * Extents are given as a template argument.
     */
    template <typename Esf, typename /*Extent*/, typename... Args>
    constexpr auto make_stage_with_extent(Args... args) GT_AUTO_RETURN(make_stage<Esf>(args...));
} // namespace gridtools
