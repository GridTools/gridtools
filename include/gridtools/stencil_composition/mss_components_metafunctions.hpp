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
        GT_META_LAZY_NAMESPACE {
            template <class>
            struct mss_split_esfs;
            template <class ExecutionEngine, class EsfSequence, class CacheSequence>
            struct mss_split_esfs<mss_descriptor<ExecutionEngine, EsfSequence, CacheSequence>> {
                GT_STATIC_ASSERT((meta::all_of<is_esf_descriptor, EsfSequence>::value), GT_INTERNAL_ERROR);
                template <class Esf>
                GT_META_DEFINE_ALIAS(
                    make_mss, meta::id, (mss_descriptor<ExecutionEngine, std::tuple<Esf>, std::tuple<>>));
                using esfs_t = GT_META_CALL(unwrap_independent, EsfSequence);
                using type = GT_META_CALL(meta::transform, (make_mss, esfs_t));
            };
        }
        GT_META_DELEGATE_TO_LAZY(mss_split_esfs, class Mss, Mss);

        template <bool Fuse, class Msses>
        struct split_mss_into_independent_esfs {
            using mms_lists_t = GT_META_CALL(meta::transform, (mss_split_esfs, Msses));
            using type = GT_META_CALL(meta::flatten, mms_lists_t);
        };

        template <class Msses>
        struct split_mss_into_independent_esfs<true, Msses> {
            using type = Msses;
        };

        template <class ExtentMap, class Axis>
        struct make_mms_components_f {
            template <class Mss>
            GT_META_DEFINE_ALIAS(apply, meta::id, (mss_components<Mss, ExtentMap, Axis>));
        };

    } // namespace mss_comonents_metafunctions_impl_

    /**
     * @brief metafunction that builds the array of mss components
     */
    template <bool Fuse,
        class Msses,
        class ExtentMap,
        class Axis,
        class SplitMsses =
            typename mss_comonents_metafunctions_impl_::split_mss_into_independent_esfs<Fuse, Msses>::type,
        class Maker = mss_comonents_metafunctions_impl_::make_mms_components_f<ExtentMap, Axis>>
    GT_META_DEFINE_ALIAS(build_mss_components_array, meta::transform, (Maker::template apply, SplitMsses));

} // namespace gridtools
