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
#include "independent_esf.hpp"

#ifndef GT_ICOSAHEDRAL_GRIDS
#include "structured_grids/esf_metafunctions.hpp"
#else
#include "icosahedral_grids/esf_metafunctions.hpp"
#endif

namespace gridtools {
    namespace esf_metafunctions_impl_ {
        template <class Esf>
        GT_META_DEFINE_ALIAS(get_items, meta::zip, (typename Esf::args_t, GT_META_CALL(esf_param_list, Esf)));

        template <intent Intent>
        struct has_intent {
            template <class Item, class Param = GT_META_CALL(meta::second, Item)>
            GT_META_DEFINE_ALIAS(apply, bool_constant, Param::intent_v == Intent);
        };

        GT_META_LAZY_NAMESPACE {
            template <class Esf>
            struct tuple_from_esf {
                using type = std::tuple<Esf>;
            };
            template <class Esfs>
            struct tuple_from_esf<independent_esf<Esfs>> {
                using type = Esfs;
            };

            template <class Item>
            struct get_out_arg : meta::lazy::first<Item> {
                using extent_t = typename meta::lazy::second<Item>::type::extent_t;
                GT_STATIC_ASSERT(extent_t::iminus::value == 0 && extent_t::iplus::value == 0 &&
                                     extent_t::jminus::value == 0 && extent_t::jplus::value == 0,
                    "Horizontal extents of the outputs of ESFs are not all empty. All outputs must have empty "
                    "(horizontal) extents");
            };
        }
        GT_META_DELEGATE_TO_LAZY(tuple_from_esf, class Esf, Esf);
        GT_META_DELEGATE_TO_LAZY(get_out_arg, class Item, Item);
    } // namespace esf_metafunctions_impl_

    /**
     *  Provide list of placeholders that corresponds to fields (temporary or not) that are written by EsfF.
     */
    template <class Esf,
        class AllItems = GT_META_CALL(esf_metafunctions_impl_::get_items, Esf),
        class WItems = GT_META_CALL(
            meta::filter, (esf_metafunctions_impl_::has_intent<intent::inout>::apply, AllItems))>
    GT_META_DEFINE_ALIAS(esf_get_w_args_per_functor, meta::transform, (meta::first, WItems));

    /**
     * Compute a list of all args specified by the user that are written into by at least one ESF
     */
    template <class Esfs,
        class ItemLists = GT_META_CALL(meta::transform, (esf_metafunctions_impl_::get_items, Esfs)),
        class AllItems = GT_META_CALL(meta::flatten, ItemLists),
        class AllRwItems = GT_META_CALL(
            meta::filter, (esf_metafunctions_impl_::has_intent<intent::inout>::apply, AllItems)),
        class AllRwArgs = GT_META_CALL(meta::transform, (meta::first, AllRwItems))>
    GT_META_DEFINE_ALIAS(compute_readwrite_args, meta::dedup, AllRwArgs);

    // Takes a list of esfs and independent_esf and produces a list of esfs, with the independent unwrapped
    template <class Esfs,
        class EsfLists = GT_META_CALL(meta::transform, (esf_metafunctions_impl_::tuple_from_esf, Esfs))>
    GT_META_DEFINE_ALIAS(unwrap_independent, meta::flatten, EsfLists);
} // namespace gridtools
