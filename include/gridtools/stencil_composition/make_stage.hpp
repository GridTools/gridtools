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

#include "../common/defs.hpp"
#include "../meta.hpp"
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
    constexpr esf_descriptor<Esf, meta::list<Args...>, Extent> make_stage(Args...) {
        static_assert(conjunction<is_plh<Args>...>::value, "Malformed make_stage");
        static_assert(sizeof...(Args) == meta::length<typename Esf::param_list>::value,
            "wrong number of arguments passed to the make_stage");
        static_assert(std::is_void<Extent>::value || is_extent<Extent>::value, "Invalid Extent type");
        return {};
    }
} // namespace gridtools
