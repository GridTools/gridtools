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
#include "mss.hpp"
#include "mss_metafunctions.hpp"

/** @file
    This file implements the metafunctions to perform data dependency analysis on a multi-stage computation (MSS).
    The idea is to assign to each placeholder used in the computation an extent that represents the values that need
    to be accessed by the stages of the computation in each iteration point. This "assignment" is done by using
    a compile time between placeholders and extents.
 */

namespace gridtools {
    namespace compute_extents_metafunctions_impl_ {
        template <class Map, class Arg>
        GT_META_DEFINE_ALIAS(lookup_extent_map,
            meta::rename,
            (enclosing_extent,
                GT_META_CALL(meta::pop_front, (GT_META_CALL(meta::mp_find, (Map, Arg, meta::list<Arg, extent<>>))))));

        template <class Map>
        struct lookup_extent_map_f {
            template <class Arg>
            GT_META_DEFINE_ALIAS(apply, lookup_extent_map, (Map, Arg));
        };

        template <intent Intent>
        struct has_intent {
            template <class Item, class Param = GT_META_CALL(meta::second, Item)>
            GT_META_DEFINE_ALIAS(apply, bool_constant, Param::intent_v == Intent);
        };

        template <class Esf>
        GT_META_DEFINE_ALIAS(get_arg_param_pairs, meta::zip, (typename Esf::args_t, GT_META_CALL(esf_param_list, Esf)));

        GT_META_LAZY_NAMESPACE {
            template <class ArgParamPair>
            struct get_out_arg : meta::lazy::first<ArgParamPair> {
                using extent_t = typename meta::lazy::second<ArgParamPair>::type::extent_t;
                GT_STATIC_ASSERT(extent_t::iminus::value == 0 && extent_t::iplus::value == 0 &&
                                     extent_t::jminus::value == 0 && extent_t::jplus::value == 0,
                    "Horizontal extents of the outputs of ESFs are not all empty. All outputs must have empty "
                    "(horizontal) extents");
            };
        }
        GT_META_DELEGATE_TO_LAZY(get_out_arg, class T, T);

        template <class Extent>
        struct make_item_f {
            template <class ArgParamPair, class Param = GT_META_CALL(meta::second, ArgParamPair)>
            GT_META_DEFINE_ALIAS(apply,
                meta::list,
                (GT_META_CALL(meta::first, ArgParamPair),
                    GT_META_CALL(sum_extent, (Extent, typename Param::extent_t))));
        };

        GT_META_LAZY_NAMESPACE {
            template <class Esf, class ExtentMap, class Extent = typename Esf::extent_t>
            struct get_esf_extent {
                using type = Extent;
            };

            template <class Esf, class ExtentMap>
            struct get_esf_extent<Esf, ExtentMap, void> {
                using arg_param_pairs_t = GT_META_CALL(get_arg_param_pairs, Esf);
                using out_args_t = GT_META_CALL(meta::transform,
                    (compute_extents_metafunctions_impl_::get_out_arg,
                        GT_META_CALL(meta::filter, (has_intent<intent::inout>::apply, arg_param_pairs_t))));
                using extents_t = GT_META_CALL(
                    meta::transform, (lookup_extent_map_f<ExtentMap>::template apply, out_args_t));
                using type = GT_META_CALL(meta::rename, (enclosing_extent, extents_t));
            };

            template <class Esf, class ExtentMap, class Extent = typename Esf::extent_t>
            struct process_esf {
                using arg_param_pairs_t = GT_META_CALL(get_arg_param_pairs, Esf);
                using new_items_t = GT_META_CALL(
                    meta::transform, (make_item_f<Extent>::template apply, arg_param_pairs_t));
                using type = GT_META_CALL(meta::lfold, (meta::mp_insert, ExtentMap, new_items_t));
            };

            template <class Esf, class ExtentMap>
            struct process_esf<Esf, ExtentMap, void> {
                using esf_extent_t = typename get_esf_extent<Esf, ExtentMap>::type;
                using in_arg_param_pairs_t = GT_META_CALL(
                    meta::filter, (has_intent<intent::in>::apply, GT_META_CALL(get_arg_param_pairs, Esf)));
                using new_items_t = GT_META_CALL(
                    meta::transform, (make_item_f<esf_extent_t>::template apply, in_arg_param_pairs_t));
                using type = GT_META_CALL(meta::lfold, (meta::mp_insert, ExtentMap, new_items_t));
            };
        }
        GT_META_DELEGATE_TO_LAZY(get_esf_extent, (class Esf, class ExtentMap), (Esf, ExtentMap));
        GT_META_DELEGATE_TO_LAZY(process_esf, (class Esf, class ExtentMap), (Esf, ExtentMap));

        template <class Esfs>
        GT_META_DEFINE_ALIAS(get_extent_map, meta::rfold, (process_esf, meta::list<>, Esfs));
    } // namespace compute_extents_metafunctions_impl_

    using compute_extents_metafunctions_impl_::get_esf_extent;
    using compute_extents_metafunctions_impl_::get_extent_map;
    using compute_extents_metafunctions_impl_::lookup_extent_map;
} // namespace gridtools
