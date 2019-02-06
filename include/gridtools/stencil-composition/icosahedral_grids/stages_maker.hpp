/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
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

        template <class Index>
        struct bind_functor_with_interval_f {
            template <class Functor>
            GT_META_DEFINE_ALIAS(apply, bind_functor_with_interval, (Functor, Index));
        };

        template <class Esf>
        struct esf_functor_f;

        GT_META_LAZY_NAMESPACE {

            template <class Esf, class Color = typename Esf::color_t::color_t>
            struct get_functors {
                static constexpr uint_t color = Color::value;
                static constexpr uint_t n_colors = Esf::location_type::n_colors::value;

                GT_STATIC_ASSERT(n_colors > 0, GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT(color < n_colors, GT_INTERNAL_ERROR);

                using before_t = GT_META_CALL(meta::repeat_c, (color, void));
                using after_t = GT_META_CALL(meta::repeat_c, (n_colors - color - 1, void));

                using type = GT_META_CALL(
                    meta::concat, (before_t, meta::list<typename Esf::template esf_function<color>>, after_t));
            };

            template <class Esf>
            struct get_functors<Esf, void> {
                using type = GT_META_CALL(meta::transform,
                    (esf_functor_f<Esf>::template apply,
                        GT_META_CALL(meta::make_indices_c, Esf::location_type::n_colors::value)));
            };

            template <class Functors, class Esf, class ExtentMap, class = void>
            struct stages_from_functors {
                using extent_t = typename get_extent_for<Esf, ExtentMap>::type;
                using type = meta::list<stage<Functors, extent_t, typename Esf::args_t, typename Esf::location_type>>;
            };
            template <class Functors, class Esf, class ExtentMap>
            struct stages_from_functors<Functors,
                Esf,
                ExtentMap,
                enable_if_t<meta::all_of<std::is_void, Functors>::value>> {
                using type = meta::list<>;
            };

            template <class Esf, class Index, class ExtentMap>
            struct stages_from_esf : stages_from_functors<GT_META_CALL(meta::transform,
                                                              (bind_functor_with_interval_f<Index>::template apply,
                                                                  typename get_functors<Esf>::type)),
                                         Esf,
                                         ExtentMap> {};

            template <class Index, class Esfs, class ExtentMap>
            struct stages_from_esf<independent_esf<Esfs>, Index, ExtentMap> {
                using stage_groups_t = GT_META_CALL(
                    meta::transform, (stages_from_esf_f<Index, ExtentMap>::template apply, Esfs));
                using stages_t = GT_META_CALL(meta::flatten, stage_groups_t);
                using type = GT_META_CALL(fuse_stages, (compound_stage, stages_t));
            };

            template <class Esf, class Color>
            struct esf_functor {
                using type = typename Esf::template esf_function<Color::value>;
            };
        }
        GT_META_DELEGATE_TO_LAZY(stages_from_esf, (class Esf, class Index, class ExtentMap), (Esf, Index, ExtentMap));
        GT_META_DELEGATE_TO_LAZY(esf_functor, (class Esf, class Color), (Esf, Color));

        template <class Index, class ExtentMap>
        struct stages_from_esf_f {
            template <class Esf>
            GT_META_DEFINE_ALIAS(apply, stages_from_esf, (Esf, Index, ExtentMap));
        };

        template <class Esf>
        struct esf_functor_f {
            template <class Color>
            GT_META_DEFINE_ALIAS(apply, esf_functor, (Esf, Color));
        };
    } // namespace _impl

    /**
     *   Transforms mss_descriptor into a sequence of stages.
     *
     * @tparam MssDescriptor -   mss descriptor
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
    template <class MssDescriptor, class ExtentMap>
    struct stages_maker;

    template <class ExecutionEngine, class Esfs, class Caches, class ExtentMap>
    struct stages_maker<mss_descriptor<ExecutionEngine, Esfs, Caches>, ExtentMap> {
        template <class LevelIndex>
        GT_META_DEFINE_ALIAS(apply, _impl::stages_from_esfs, (Esfs, LevelIndex, ExtentMap));
    };
} // namespace gridtools
