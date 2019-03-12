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

#include <boost/mpl/pair.hpp>

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
        GT_META_LAZY_NAMESPACE {
            template <class Esf>
            struct tuple_from_esf {
                using type = std::tuple<Esf>;
            };
            template <class Esfs>
            struct tuple_from_esf<independent_esf<Esfs>> {
                using type = Esfs;
            };
        }
        GT_META_DELEGATE_TO_LAZY(tuple_from_esf, class Esf, Esf);

        template <class Pair>
        struct horizotal_extent_is_zero
            : bool_constant<Pair::second::iminus::value == 0 && Pair::second::iplus::value == 0 &&
                            Pair::second::jminus::value == 0 && Pair::second::jplus::value == 0> {};

        template <class Esf>
        GT_META_DEFINE_ALIAS(get_items, meta::zip, (GT_META_CALL(esf_param_list, Esf), typename Esf::args_t));

        template <intent Intent>
        struct has_intent {
            template <class Item, class Param = GT_META_CALL(meta::first, Item)>
            GT_META_DEFINE_ALIAS(apply, bool_constant, Param::intent_v == Intent);
        };

        template <class Item,
            class Param = GT_META_CALL(meta::first, Item),
            class Arg = GT_META_CALL(meta::second, Item)>
        GT_META_DEFINE_ALIAS(get_arg_extent_pair, boost::mpl::pair, (Arg, typename Param::extent_t));
    } // namespace esf_metafunctions_impl_

    /**
     *  Provide list of placeholders that corresponds to fields (temporary or not) that are written by EsfF.
     */
    template <class Esf,
        class AllItems = GT_META_CALL(esf_metafunctions_impl_::get_items, Esf),
        class WItems = GT_META_CALL(
            meta::filter, (esf_metafunctions_impl_::has_intent<intent::inout>::apply, AllItems))>
    GT_META_DEFINE_ALIAS(esf_get_w_args_per_functor, meta::transform, (meta::second, WItems));

    /**
     * Provide a tuple of pairs of placeholders and extents that corresponds to fields that are written by Esf.
     */
    template <class Esf,
        class AllItems = GT_META_CALL(esf_metafunctions_impl_::get_items, Esf),
        class WItems = GT_META_CALL(
            meta::filter, (esf_metafunctions_impl_::has_intent<intent::inout>::apply, AllItems))>
    GT_META_DEFINE_ALIAS(
        esf_get_w_per_functor, meta::transform, (esf_metafunctions_impl_::get_arg_extent_pair, WItems));

    /**
     * Provide a tuple of pairs of placeholders and extents that corresponds to fields that are read by Esf.
     */
    template <class Esf,
        class AllItems = GT_META_CALL(esf_metafunctions_impl_::get_items, Esf),
        class RItems = GT_META_CALL(meta::filter, (esf_metafunctions_impl_::has_intent<intent::in>::apply, AllItems))>
    GT_META_DEFINE_ALIAS(
        esf_get_r_per_functor, meta::transform, (esf_metafunctions_impl_::get_arg_extent_pair, RItems));

    /**
     * Compute a list of all args specified by the user that are written into by at least one ESF
     */
    template <class Esfs,
        class ItemLists = GT_META_CALL(meta::transform, (esf_metafunctions_impl_::get_items, Esfs)),
        class AllItems = GT_META_CALL(meta::flatten, ItemLists),
        class AllRwItems = GT_META_CALL(
            meta::filter, (esf_metafunctions_impl_::has_intent<intent::inout>::apply, AllItems)),
        class AllRwArgs = GT_META_CALL(meta::transform, (meta::second, AllRwItems))>
    GT_META_DEFINE_ALIAS(compute_readwrite_args, meta::dedup, AllRwArgs);

    /**
      Given an array of pairs (placeholder, extent) checks if all extents are the same and equal to the extent
      passed in
     */
    template <class Pairs>
    struct check_all_horizotal_extents_are_zero
        : is_sequence_of<Pairs, esf_metafunctions_impl_::horizotal_extent_is_zero> {};

    // Takes a list of esfs and independent_esf and produces a list of esfs, with the independent unwrapped
    template <class Esfs,
        class EsfLists = GT_META_CALL(meta::transform, (esf_metafunctions_impl_::tuple_from_esf, Esfs))>
    GT_META_DEFINE_ALIAS(unwrap_independent, meta::flatten, EsfLists);

} // namespace gridtools
