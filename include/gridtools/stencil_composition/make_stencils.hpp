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

#include "../common/defs.hpp"
#include "../meta.hpp"
#include "caches/cache_traits.hpp"
#include "esf.hpp"
#include "mss.hpp"

namespace gridtools {
    template <class ExecutionEngine, class... CacheItems, class... Esfs>
    constexpr auto make_multistage(ExecutionEngine, cache_map<CacheItems...>, Esfs...) {
        static_assert(is_execution_engine<ExecutionEngine>::value,
            "The first argument passed to make_multistage must be the execution engine (e.g. execute::forward(), "
            "execute::backward(), execute::parallel()");
        static_assert(conjunction<is_esf_descriptor<Esfs>...>::value, "wrong make_multistage params.");

        return mss_descriptor<ExecutionEngine,
            meta::list<Esfs...>,
            meta::list<
#ifndef GT_DISABLE_CACHING
                CacheItems...
#endif
                >>{};
    }

    template <class ExecutionEngine, class... Esfs>
    constexpr auto make_multistage(ExecutionEngine ee, Esfs... esfs) {
        return make_multistage(ee, cache_map<>(), esfs...);
    }

} // namespace gridtools
