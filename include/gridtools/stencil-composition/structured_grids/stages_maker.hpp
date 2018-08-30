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
#include "../../common/generic_metafunctions/meta.hpp"
#include "../../common/generic_metafunctions/type_traits.hpp"
#include "../bind_functor_with_interval.hpp"
#include "../compute_extents_metafunctions.hpp"
#include "../independent_esf.hpp"
#include "../mss.hpp"
#include "../reductions/reduction_descriptor.hpp"
#include "./esf.hpp"
#include "./stage.hpp"

namespace gridtools {

    namespace _impl {
        template <class Index, class ExtentMap, size_t RepeatFactor>
        struct stages_from_esf_f;

        template <class Esfs, class Index, class ExtentMap, size_t RepeatFactor>
        GT_META_DEFINE_ALIAS(stages_from_esfs,
            meta::filter,
            (meta::not_<meta::is_empty>::apply,
                GT_META_CALL(
                    meta::transform, (stages_from_esf_f<Index, ExtentMap, RepeatFactor>::template apply, Esfs))));

        GT_META_LAZY_NAMESPASE {
            template <class Functor, class Esf, class ExtentMap, size_t RepeatFactor>
            struct stages_from_functor {
                using extent_t = typename get_extent_for<Esf, ExtentMap>::type;
                using type = meta::list<regular_stage<Functor, extent_t, typename Esf::args_t, RepeatFactor>>;
            };
            template <class Esf, class ExtentMap, size_t RepeatFactor>
            struct stages_from_functor<void, Esf, ExtentMap, RepeatFactor> {
                using type = meta::list<>;
            };

            template <class Esf, class Index, class ExtentMap, size_t RepeatFactor>
            struct stages_from_esf
                : stages_from_functor<GT_META_CALL(bind_functor_with_interval, (typename Esf::esf_function, Index)),
                      Esf,
                      ExtentMap,
                      RepeatFactor> {};

            template <class Index, class Esfs, class ExtentMap, size_t RepeatFactor>
            struct stages_from_esf<independent_esf<Esfs>, Index, ExtentMap, RepeatFactor> {
                using type = GT_META_CALL(
                    meta::flatten, (GT_META_CALL(stages_from_esfs, (Esfs, Index, ExtentMap, RepeatFactor))));
            };

            template <class Functor, class Esf, class BinOp, class ExtentMap>
            struct reduction_stages_from_functor {
                using extent_t = typename reduction_get_extent_for<Esf, ExtentMap>::type;
                using type = meta::list<meta::list<reduction_stage<Functor, extent_t, typename Esf::args_t, BinOp>>>;
            };
            template <class Esf, class BinOp, class ExtentMap>
            struct reduction_stages_from_functor<void, Esf, BinOp, ExtentMap> {
                using type = meta::list<>;
            };
            template <class Esf, class Index, class BinOp, class ExtentMap>
            struct reduction_stages : reduction_stages_from_functor<GT_META_CALL(bind_functor_with_interval,
                                                                        (typename Esf::esf_function, Index)),
                                          Esf,
                                          BinOp,
                                          ExtentMap> {};
        }
        GT_META_DELEGATE_TO_LAZY(stages_from_esf,
            (class Esf, class Index, class ExtentMap, size_t RepeatFactor),
            (Esf, Index, ExtentMap, RepeatFactor));
        GT_META_DELEGATE_TO_LAZY(
            reduction_stages, (class Esf, class Index, class BinOp, class ExtentMap), (Esf, Index, BinOp, ExtentMap));

        template <class Index, class ExtentMap, size_t RepeatFactor>
        struct stages_from_esf_f {
            template <class Esf>
            GT_META_DEFINE_ALIAS(apply, stages_from_esf, (Esf, Index, ExtentMap, RepeatFactor));
        };
    } // namespace _impl

    template <class MssDescriptor, class ExtentMap, size_t RepeatFactor>
    struct stages_maker;

    template <class ExecutionEngine, class Esfs, class Caches, class ExtentMap, size_t RepeatFactor>
    struct stages_maker<mss_descriptor<ExecutionEngine, Esfs, Caches>, ExtentMap, RepeatFactor> {
        template <class LevelIndex>
        GT_META_DEFINE_ALIAS(apply, _impl::stages_from_esfs, (Esfs, LevelIndex, ExtentMap, RepeatFactor));
    };

    template <class ReductionType,
        class BinOp,
        template <class...> class L,
        class Esf,
        class ExtentMap,
        size_t RepeatFactor>
    struct stages_maker<reduction_descriptor<ReductionType, BinOp, L<Esf>>, ExtentMap, RepeatFactor> {
        GRIDTOOLS_STATIC_ASSERT(RepeatFactor == 1, "Expandable parameters for reductions are not supported");
        template <class LevelIndex>
        GT_META_DEFINE_ALIAS(apply, _impl::reduction_stages, (Esf, LevelIndex, BinOp, ExtentMap));
    };

} // namespace gridtools
