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

        template <template <uint_t> class ColoredFunctor, class Index>
        struct bind_colored_functor_with_interval {
            template <uint_t Color>
            using apply = GT_META_CALL(bind_functor_with_interval, (ColoredFunctor<Color>, Index));
        };

        GT_META_LAZY_NAMESPASE {
            template <template <uint_t> class Functor,
                class Esf,
                class ExtentMap,
                size_t RepeatFactor,
                class Color = typename Esf::color_t::color_t,
                class FirstFunctor = Functor<0>>
            struct stages_from_functor {
                static constexpr uint_t color = Color::value;
                using extent_t = typename get_extent_for<Esf, ExtentMap>::type;
                using type = meta::list<color_specific_stage<color,
                    Functor<color>,
                    extent_t,
                    typename Esf::args_t,
                    typename Esf::location_type,
                    RepeatFactor>>;
            };
            template <template <uint_t> class Functor, class Esf, class ExtentMap, size_t RepeatFactor>
            struct stages_from_functor<Functor, Esf, ExtentMap, RepeatFactor, notype, void> {
                using type = meta::list<>;
            };
            template <template <uint_t> class Functor, class Esf, class ExtentMap, size_t RepeatFactor, class Color>
            struct stages_from_functor<Functor, Esf, ExtentMap, RepeatFactor, Color, void> {
                using type = meta::list<>;
            };
            template <template <uint_t> class Functor,
                class Esf,
                class ExtentMap,
                size_t RepeatFactor,
                class FirstFunctor>
            struct stages_from_functor<Functor, Esf, ExtentMap, RepeatFactor, notype, FirstFunctor> {
                using extent_t = typename get_extent_for<Esf, ExtentMap>::type;
                using type = meta::list<all_colors_stage<Functor,
                    extent_t,
                    typename Esf::args_t,
                    typename Esf::location_type,
                    RepeatFactor>>;
            };
            template <class Esf, class Index, class ExtentMap, size_t RepeatFactor>
            struct stages_from_esf
                : stages_from_functor<
                      bind_colored_functor_with_interval<Esf::template esf_function, Index>::template apply,
                      Esf,
                      ExtentMap,
                      RepeatFactor> {};

            template <class Index, class Esfs, class ExtentMap, size_t RepeatFactor>
            struct stages_from_esf<independent_esf<Esfs>, Index, ExtentMap, RepeatFactor> {
                using type = GT_META_CALL(
                    meta::flatten, (GT_META_CALL(stages_from_esfs, (Esfs, Index, ExtentMap, RepeatFactor))));
            };
        }
        GT_META_DELEGATE_TO_LAZY(stages_from_esf,
            (class Esf, class Index, class ExtentMap, size_t RepeatFactor),
            (Esf, Index, ExtentMap, RepeatFactor));

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
} // namespace gridtools
