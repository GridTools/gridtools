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

#include "../meta.hpp"
#include "esf_metafunctions.hpp"

namespace gridtools {
    namespace extract_placeholders_impl_ {
        template <class Trees, template <class...> class FlattenTree>
        using flatten_trees = meta::flatten<meta::transform<FlattenTree, Trees>>;

        // Extract args from ESF.
        template <class Esf>
        using get_args = typename Esf::args_t;

        // Extract ESFs from an MSS.
        template <class Mss>
        using get_esfs = unwrap_independent<typename Mss::esf_sequence_t>;
    } // namespace extract_placeholders_impl_

    template <class Mss, class Esfs = unwrap_independent<typename Mss::esf_sequence_t>>
    using extract_placeholders_from_mss =
        meta::dedup<extract_placeholders_impl_::flatten_trees<Esfs, extract_placeholders_impl_::get_args>>;

    /// Takes a type list of MSS descriptions and returns deduplicated type list of all placeholders
    /// that are used in the given msses.
    template <class Msses>
    using extract_placeholders_from_msses = meta::dedup<extract_placeholders_impl_::flatten_trees<
        extract_placeholders_impl_::flatten_trees<Msses, extract_placeholders_impl_::get_esfs>,
        extract_placeholders_impl_::get_args>>;
} // namespace gridtools
