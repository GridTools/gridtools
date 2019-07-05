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
#include "accessor_intent.hpp"

#ifndef GT_ICOSAHEDRAL_GRIDS
#include "structured_grids/esf_metafunctions.hpp"
#else
#include "icosahedral_grids/esf_metafunctions.hpp"
#endif

namespace gridtools {
    namespace esf_metafunctions_impl_ {
        template <class Esf>
        using get_items = meta::zip<typename Esf::args_t, esf_param_list<Esf>>;

        template <intent Intent>
        struct has_intent {
            template <class Item, class Param = meta::second<Item>>
            using apply = bool_constant<Param::intent_v == Intent>;
        };

        namespace lazy {
            template <class Item>
            struct get_out_arg : meta::lazy::first<Item> {
                using extent_t = typename meta::lazy::second<Item>::type::extent_t;
                GT_STATIC_ASSERT(extent_t::iminus::value == 0 && extent_t::iplus::value == 0 &&
                                     extent_t::jminus::value == 0 && extent_t::jplus::value == 0,
                    "Horizontal extents of the outputs of ESFs are not all empty. All outputs must have empty "
                    "(horizontal) extents");
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(get_out_arg, class Item, Item);
    } // namespace esf_metafunctions_impl_

    /**
     *  Provide list of placeholders that corresponds to fields (temporary or not) that are written by EsfF.
     */
    template <class Esf,
        class AllItems = esf_metafunctions_impl_::get_items<Esf>,
        class WItems = meta::filter<esf_metafunctions_impl_::has_intent<intent::inout>::apply, AllItems>>
    using esf_get_w_args_per_functor = meta::transform<meta::first, WItems>;

    /**
     * Compute a list of all args specified by the user that are written into by at least one ESF
     */
    template <class Esfs,
        class ItemLists = meta::transform<esf_metafunctions_impl_::get_items, Esfs>,
        class AllItems = meta::flatten<ItemLists>,
        class AllRwItems = meta::filter<esf_metafunctions_impl_::has_intent<intent::inout>::apply, AllItems>,
        class AllRwArgs = meta::transform<meta::first, AllRwItems>>
    using compute_readwrite_args = meta::dedup<AllRwArgs>;

    template <class Esfs,
        class ItemLists = meta::transform<esf_metafunctions_impl_::get_items, Esfs>,
        class AllItems = meta::flatten<ItemLists>,
        class AllRwItems = meta::filter<esf_metafunctions_impl_::has_intent<intent::inout>::apply, AllItems>,
        class RwArgs = meta::dedup<meta::transform<meta::first, AllRwItems>>,
        class Args = meta::dedup<meta::transform<meta::first, AllItems>>>
    using compute_readonly_args =
        meta::transform<meta::not_<meta::curry<meta::st_contains, RwArgs>::template apply>::template apply, Args>;
} // namespace gridtools
