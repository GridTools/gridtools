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

#include "../common/defs.hpp"
#include "../meta.hpp"
#include "caches/cache_traits.hpp"
#include "esf_fwd.hpp"
#include "mss.hpp"

namespace gridtools {
    /**
     *  Function to create a Multistage Stencil that can then be executed
     */
    template <class ExecutionEngine, class... Params>
    constexpr auto make_multistage(ExecutionEngine, Params...) {
        GT_STATIC_ASSERT(is_execution_engine<ExecutionEngine>::value,
            "The first argument passed to make_multistage must be the execution engine (e.g. execute::forward(), "
            "execute::backward(), execute::parallel()");

        GT_STATIC_ASSERT(conjunction<meta::is_list<Params>...>::value, "wrong make_multistage params.");

        using params_t = meta::concat<std::tuple<>, Params...>;
        using esfs_t = meta::filter<is_esf_descriptor, params_t>;
        using caches_t = meta::filter<is_cache, params_t>;

        GT_STATIC_ASSERT(meta::length<esfs_t>::value + meta::length<caches_t>::value == meta::length<params_t>::value,
            "wrong set of mss parameters passed to make_multistage construct.\n"
            "Check that arguments passed are either :\n"
            " * caches from define_caches(...) construct or\n"
            " * esf descriptors from make_stage(...) or make_independent(...)");

#ifdef GT_DISABLE_CACHING
        using effective_caches_t = std::tuple<>;
#else
        using effective_caches_t = caches_t;
#endif
        return mss_descriptor<ExecutionEngine, esfs_t, effective_caches_t>{};
    }

    // Deprecated.
    template <class... EsfTups>
    constexpr meta::concat<EsfTups...> make_independent(EsfTups...) {
        GT_STATIC_ASSERT((conjunction<meta::all_of<is_esf_descriptor, EsfTups>...>::value),
            "make_independent arguments should be results of make_stage.");
        return {};
    }

} // namespace gridtools
