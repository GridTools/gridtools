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

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../meta.hpp"
#include "../bind_functor_with_interval.hpp"
#include "../compute_extents_metafunctions.hpp"
#include "../fuse_stages.hpp"
#include "../independent_esf.hpp"
#include "../mss.hpp"
#include "./esf.hpp"
#include "./stage.hpp"

namespace gridtools {

    namespace _impl {
        template <class Index, class ExtentMap>
        struct stages_from_esf_f;

        template <class Esfs, class Index, class ExtentMap>
        GT_META_DEFINE_ALIAS(stages_from_esfs,
            meta::filter,
            (meta::not_<meta::is_empty>::apply,
                GT_META_CALL(meta::transform, (stages_from_esf_f<Index, ExtentMap>::template apply, Esfs))));

        GT_META_LAZY_NAMESPACE {
            template <class Functor, class Esf, class ExtentMap>
            struct stages_from_functor {
                using extent_t = typename get_extent_for<Esf, ExtentMap>::type;
                using type = meta::list<regular_stage<Functor, extent_t, typename Esf::args_t>>;
            };
            template <class Esf, class ExtentMap>
            struct stages_from_functor<void, Esf, ExtentMap> {
                using type = meta::list<>;
            };

            template <class Esf, class Index, class ExtentMap>
            struct stages_from_esf
                : stages_from_functor<GT_META_CALL(bind_functor_with_interval, (typename Esf::esf_function_t, Index)),
                      Esf,
                      ExtentMap> {};

            template <class Index, class Esfs, class ExtentMap>
            struct stages_from_esf<independent_esf<Esfs>, Index, ExtentMap> {
                using stage_groups_t = GT_META_CALL(
                    meta::transform, (stages_from_esf_f<Index, ExtentMap>::template apply, Esfs));
                using stages_t = GT_META_CALL(meta::flatten, stage_groups_t);
                using type = GT_META_CALL(fuse_stages, (compound_stage, stages_t));
            };
        }
        GT_META_DELEGATE_TO_LAZY(stages_from_esf, (class Esf, class Index, class ExtentMap), (Esf, Index, ExtentMap));

        template <class Index, class ExtentMap>
        struct stages_from_esf_f {
            template <class Esf>
            GT_META_DEFINE_ALIAS(apply, stages_from_esf, (Esf, Index, ExtentMap));
        };
    } // namespace _impl

    /**
     *   Transforms `mss_descriptor` into a sequence of stages.
     *
     * @tparam Descriptor -   mss descriptor
     * @tparam ExtentMap -    a compile time map that maps placeholders to computed extents.
     *                        `stages_maker` uses ExtentMap parameter in an opaque way -- it just delegates it to
     *                        `get_extent_for` when it is needed.
     *
     *   This metafunction returns another metafunction (i.e. has nested `apply` metafunction) that accepts
     *   a single argument that has to be a level_index and returns the stages (classes that model Stage concept)
     *   The returned stages are organized as a type list of type lists. The inner lists represent the stages that are
     *   independent on each other. The outer list represents the sequence that should be executed in order.
     *   It is guarantied that all inner lists are not empty.
     *   Examples of valid return types:
     *      list<> -  no stages should be executed for the given interval level
     *      list<list<stage1>> - a singe stage to execute
     *      list<list<stage1>, list<stage2>> - two stages should be executed in the given order
     *      list<list<stage1, stage2>> - two stages should be executed in any order or in paralel
     *      list<list<stage1>, list<stage2, stage3>, list<stage4>> - an order of execution can be
     *         either 1,2,3,4 or 1,3,2,4
     *
     *   Note that the collection of stages is calculated for the provided level_index. It can happen that same mss
     *   produces different stages for the different level indices because of interval overloads. If for two level
     *   indices all elementary functors should be called with the same correspondent intervals, the returned stages
     *   will be exactly the same.
     *
     *   TODO(anstaf): unit test!!!
     */
    template <class Descriptor, class ExtentMap>
    struct stages_maker;

    template <class ExecutionEngine, class Esfs, class Caches, class ExtentMap>
    struct stages_maker<mss_descriptor<ExecutionEngine, Esfs, Caches>, ExtentMap> {
        template <class LevelIndex>
        GT_META_DEFINE_ALIAS(apply, _impl::stages_from_esfs, (Esfs, LevelIndex, ExtentMap));
    };
} // namespace gridtools
