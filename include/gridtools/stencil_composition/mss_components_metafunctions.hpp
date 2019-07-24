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
#include "esf_metafunctions.hpp"
#include "mss.hpp"
#include "mss_components.hpp"

namespace gridtools {
    namespace mss_comonents_metafunctions_impl_ {
        namespace lazy {
            template <class>
            struct mss_split_esfs;
            template <class ExecutionEngine, class EsfSequence, class CacheSequence>
            struct mss_split_esfs<mss_descriptor<ExecutionEngine, EsfSequence, CacheSequence>> {
                GT_STATIC_ASSERT((meta::all_of<is_esf_descriptor, EsfSequence>::value), GT_INTERNAL_ERROR);
                template <class Esf>
                using make_mss = mss_descriptor<ExecutionEngine, std::tuple<Esf>, CacheSequence>;

                using type = meta::transform<make_mss, EsfSequence>;
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(mss_split_esfs, class Mss, Mss);

        template <class Msses>
        struct split_mss_into_independent_esfs {
            using mms_lists_t = meta::transform<mss_split_esfs, Msses>;
            using type = meta::flatten<mms_lists_t>;
        };

        template <class ExtentMap, class Axis>
        struct make_mms_components_f {
            template <class Mss>
            using apply = mss_components<Mss, ExtentMap, Axis>;
        };

    } // namespace mss_comonents_metafunctions_impl_

    /**
     * @brief metafunction that builds the array of mss components
     */
    template <class Msses,
        class ExtentMap,
        class Axis,
        class SplitMsses = typename mss_comonents_metafunctions_impl_::split_mss_into_independent_esfs<Msses>::type,
        class Maker = mss_comonents_metafunctions_impl_::make_mms_components_f<ExtentMap, Axis>>
    using build_mss_components_array = meta::transform<Maker::template apply, SplitMsses>;

} // namespace gridtools
