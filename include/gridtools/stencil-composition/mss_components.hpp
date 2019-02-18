/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../meta.hpp"
#include "compute_extents_metafunctions.hpp"
#include "esf_metafunctions.hpp"
#include "make_loop_intervals.hpp"
#include "mss.hpp"
#include "mss_metafunctions.hpp"
#include "stages_maker.hpp"

namespace gridtools {

    namespace mss_comonents_impl_ {
        template <class Esf>
        GT_META_DEFINE_ALIAS(esf_produce_temporary,
            bool_constant,
            !boost::mpl::empty<typename esf_get_w_temps_per_functor<Esf>::type>::value);

        template <class ExtentMap>
        struct get_extent_f {
            template <class Esf>
            GT_META_DEFINE_ALIAS(apply, meta::id, (typename get_extent_for<Esf, ExtentMap>::type));
        };

        template <class Esfs,
            class ExtentMap,
            class TmpEsfs = GT_META_CALL(meta::filter, (esf_produce_temporary, Esfs)),
            class Extents = GT_META_CALL(meta::transform, (get_extent_f<ExtentMap>::template apply, TmpEsfs))>
        GT_META_DEFINE_ALIAS(get_max_extent_for_tmp, meta::rename, (enclosing_extent, Extents));
    } // namespace mss_comonents_impl_

    /**
     * @brief the mss components contains meta data associated to a mss descriptor.
     * All derived metadata is computed in this class
     * @tparam MssDescriptor the mss descriptor
     * @tparam ExtentSizes the extent sizes of all the ESFs in this mss
     * @tparam RepeatFunctor the length of the chunks for expandable parameters
     */
    template <typename MssDescriptor, typename ExtentMap, typename Axis>
    struct mss_components {
        GT_STATIC_ASSERT(is_mss_descriptor<MssDescriptor>::value, GT_INTERNAL_ERROR);
        typedef MssDescriptor mss_descriptor_t;

        typedef typename MssDescriptor::execution_engine_t execution_engine_t;

        /** Collect all esf nodes in the the multi-stage descriptor. Recurse into independent
            esf structs. Independent functors are listed one after the other.*/
        using linear_esf_t = GT_META_CALL(unwrap_independent, typename MssDescriptor::esf_sequence_t);

        using extent_map_t = ExtentMap;

        typedef typename MssDescriptor::cache_sequence_t cache_sequence_t;

        // For historical reasons the user provided axis interval is stripped by one level from the right to produce
        // the interval that will be used for actual computation.
        // TODO(anstaf): fix this ugly convention
        using default_interval_t = interval<typename Axis::FromLevel,
            GT_META_CALL(index_to_level, typename level_to_index<typename Axis::ToLevel>::prior)>;

        // calculate loop intervals and order them according to the execution policy.
        using loop_intervals_t = GT_META_CALL(order_loop_intervals,
            (execution_engine_t,
                GT_META_CALL(make_loop_intervals,
                    (stages_maker<MssDescriptor, ExtentMap>::template apply, default_interval_t))));
    };

    template <typename T>
    GT_META_DEFINE_ALIAS(is_mss_components, meta::is_instantiation_of, (mss_components, T));

    template <class MssComponents>
    GT_META_DEFINE_ALIAS(get_max_extent_for_tmp_from_mss_components,
        mss_comonents_impl_::get_max_extent_for_tmp,
        (typename MssComponents::linear_esf_t, typename MssComponents::extent_map_t));
} // namespace gridtools
