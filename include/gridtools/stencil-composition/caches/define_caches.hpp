/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <tuple>

#include "../../common/defs.hpp"
#include "../../meta/concat.hpp"
#include "../../meta/logical.hpp"
#include "../../meta/macros.hpp"
#include "../../meta/type_traits.hpp"
#include "./cache_traits.hpp"

namespace gridtools {

    /**
     * function that captures the list of caches provided by the user for a stencil
     */
    template <class... CacheSequences>
    GT_META_CALL(meta::concat, (std::tuple<>, CacheSequences...))
    define_caches(CacheSequences...) {
        // the call to define_caches might gets a variadic list of cache sequences as input
        // (e.g., define_caches(cache<IJ, local>(p_flx(), p_fly()), cache<K, fill>(p_in())); ).
        GT_STATIC_ASSERT((conjunction<meta::all_of<is_cache, CacheSequences>...>::value),
            "Error: did not provide a sequence of caches to define_caches syntax");
        return {};
    }

} // namespace gridtools
