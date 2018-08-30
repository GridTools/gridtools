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

        template <class Index>
        struct bind_functor_with_interval_f {
            template <class Functor>
            GT_META_DEFINE_ALIAS(apply, bind_functor_with_interval, (Functor, Index));
        };

        template <class Esf>
        struct esf_functor_f;

        GT_META_LAZY_NAMESPASE {

            template <class Esf, class Color = typename Esf::color_t::color_t>
            struct get_functors {
                static constexpr uint_t color = Color::value;
                static constexpr uint_t n_colors = Esf::location_type::n_colors::value;

                GRIDTOOLS_STATIC_ASSERT(n_colors > 0, GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT(color < n_colors, GT_INTERNAL_ERROR);

                using before_t = GT_META_CALL(meta::repeat_c, (color, void));
                using after_t = GT_META_CALL(meta::repeat_c, (n_colors - color - 1, void));

                using type = GT_META_CALL(
                    meta::concat, (before_t, meta::list<typename Esf::template esf_function<color>>, after_t));
            };

            template <class Esf>
            struct get_functors<Esf, notype> {
                using type = GT_META_CALL(meta::transform,
                    (esf_functor_f<Esf>::template apply,
                        GT_META_CALL(meta::make_indices_c, Esf::location_type::n_colors::value)));
            };

            template <class Functors, class Esf, class ExtentMap, size_t RepeatFactor, class = void>
            struct stages_from_functors {
                using extent_t = typename get_extent_for<Esf, ExtentMap>::type;
                using type = meta::list<
                    stage<Functors, extent_t, typename Esf::args_t, typename Esf::location_type, RepeatFactor>>;
            };
            template <class Functors, class Esf, class ExtentMap, size_t RepeatFactor>
            struct stages_from_functors<Functors,
                Esf,
                ExtentMap,
                RepeatFactor,
                enable_if_t<meta::all_of<std::is_void, Functors>::value>> {
                using type = meta::list<>;
            };

            template <class Esf, class Index, class ExtentMap, size_t RepeatFactor>
            struct stages_from_esf : stages_from_functors<GT_META_CALL(meta::transform,
                                                              (bind_functor_with_interval_f<Index>::template apply,
                                                                  typename get_functors<Esf>::type)),
                                         Esf,
                                         ExtentMap,
                                         RepeatFactor> {};

            template <class Index, class Esfs, class ExtentMap, size_t RepeatFactor>
            struct stages_from_esf<independent_esf<Esfs>, Index, ExtentMap, RepeatFactor> {
                using type = GT_META_CALL(
                    meta::flatten, (GT_META_CALL(stages_from_esfs, (Esfs, Index, ExtentMap, RepeatFactor))));
            };

            template <class Esf, class Color>
            struct esf_functor {
                using type = typename Esf::template esf_function<Color::value>;
            };
        }
        GT_META_DELEGATE_TO_LAZY(stages_from_esf,
            (class Esf, class Index, class ExtentMap, size_t RepeatFactor),
            (Esf, Index, ExtentMap, RepeatFactor));
        GT_META_DELEGATE_TO_LAZY(esf_functor, (class Esf, class Color), (Esf, Color));

        template <class Index, class ExtentMap, size_t RepeatFactor>
        struct stages_from_esf_f {
            template <class Esf>
            GT_META_DEFINE_ALIAS(apply, stages_from_esf, (Esf, Index, ExtentMap, RepeatFactor));
        };

        template <class Esf>
        struct esf_functor_f {
            template <class Color>
            GT_META_DEFINE_ALIAS(apply, esf_functor, (Esf, Color));
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
