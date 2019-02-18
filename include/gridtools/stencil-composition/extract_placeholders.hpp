/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "../meta.hpp"
#include "esf_metafunctions.hpp"

namespace gridtools {
    namespace extract_placeholders_impl_ {
        template <class Trees, template <class...> class FlattenTree>
        GT_META_DEFINE_ALIAS(flatten_trees, meta::flatten, (GT_META_CALL(meta::transform, (FlattenTree, Trees))));

        // Extract args from ESF.
        template <class Esf>
        GT_META_DEFINE_ALIAS(get_args, meta::id, typename Esf::args_t);

        // Extract ESFs from an MSS.
        template <class Mss>
        GT_META_DEFINE_ALIAS(get_esfs, unwrap_independent, typename Mss::esf_sequence_t);
    } // namespace extract_placeholders_impl_

    template <class Mss, class Esfs = GT_META_CALL(unwrap_independent, typename Mss::esf_sequence_t)>
    GT_META_DEFINE_ALIAS(extract_placeholders_from_mss,
        meta::dedup,
        (GT_META_CALL(extract_placeholders_impl_::flatten_trees, (Esfs, extract_placeholders_impl_::get_args))));

    /// Takes a type list of MSS descriptions and returns deduplicated type list of all placeholders
    /// that are used in the given msses.
    template <class Msses>
    GT_META_DEFINE_ALIAS(extract_placeholders_from_msses,
        meta::dedup,
        (GT_META_CALL(extract_placeholders_impl_::flatten_trees,
            (GT_META_CALL(extract_placeholders_impl_::flatten_trees, (Msses, extract_placeholders_impl_::get_esfs)),
                extract_placeholders_impl_::get_args))));
} // namespace gridtools
