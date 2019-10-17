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

#include "../../meta.hpp"
#include "../arg.hpp"
#include "cache_definitions.hpp"
#include "cache_traits.hpp"

namespace gridtools {

    template <class Type, class... IOPolicies, class... Plhs>
    cache_map<cache_info<Plhs, meta::list<Type>, meta::list<IOPolicies...>>...> cache(Plhs...) {
        static_assert(conjunction<is_plh<Plhs>...>::value, "argument passed to cache is not of the right arg<> type");
        return {};
    }

    /**
     * function that captures the list of caches provided by the user for a stencil
     */
    template <class... CacheMaps>
    meta::concat<CacheMaps...> define_caches(CacheMaps...) {
        static_assert(conjunction<meta::is_instantiation_of<cache_map, CacheMaps>...>::value,
            "Error: did not provide a sequence of caches to define_caches syntax");
        return {};
    }

} // namespace gridtools
