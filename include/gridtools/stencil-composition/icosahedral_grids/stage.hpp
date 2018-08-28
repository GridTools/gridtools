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

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/generic_metafunctions/meta.hpp"
#include "../../common/generic_metafunctions/type_traits.hpp"
#include "../../common/host_device.hpp"
#include "../arg.hpp"
#include "../extent.hpp"
#include "./iterate_domain_expandable_parameters.hpp"
#include "./iterate_domain_remapper.hpp"

namespace gridtools {

    namespace _impl {
        template <class EnclosedExtent, class ItDomain>
        struct exec_stage_f {
            ItDomain &m_domain;

            template <class Extent, class Stage>
            GT_FUNCTION void operator()(meta::list<Stage, Extent>) const {
                if (m_domain.template is_thread_in_domain<Extent>())
                    Stage::exec(m_domain);
            }
            template <class Extent, class Stage>
            GT_FUNCTION void operator()(meta::list<Stage, EnclosedExtent>) const {
                Stage::exec(m_domain);
            }
        };

        template <uint_t Color, class EnclosedExtent, class ItDomain>
        struct exec_stage_for_color_f {
            ItDomain &m_domain;

            template <class Extent, class Stage>
            GT_FUNCTION void operator()(meta::list<Stage, Extent>) const {
                if (m_domain.template is_thread_in_domain<Extent>())
                    Stage::template exec<Color>(m_domain);
            }
            template <class Extent, class Stage>
            GT_FUNCTION void operator()(meta::list<Stage, EnclosedExtent>) const {
                Stage::template exec<Color>(m_domain);
            }
        };

        template <class Functor, class Eval>
        struct call_do {
            Eval *m_eval;
            template <class Index>
            GT_FUNCTION void operator()(Index) const {
                using eval_t = iterate_domain_expandable_parameters<Eval, Index::value + 1>;
                Functor::template Do<eval_t &>(*reinterpret_cast<eval_t *>(m_eval));
            }
        };

    } // namespace _impl

    template <class... Stages>
    struct merged_stage {
        using extents_t = meta::list<typename Stages::extent_t...>;
        using stages_t = meta::list<Stages...>;

        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_extent, extents_t>::value), GT_INTERNAL_ERROR);

        using extent_t = GT_META_CALL(enclosing_extent, typename Stages::extent_t...);

        template <uint_t C>
        struct contains_color : disjunction<typename Stages::template containts_color<C>...> {};

        template <uint_t Color, class ItDomain>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            gridtools::for_each<GT_META_CALL(meta::zip, (stages_t, extents_t))>(
                _impl::exec_stage_for_color_f<Color, extent_t, ItDomain>{it_domain});
        }

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            gridtools::for_each<GT_META_CALL(meta::zip, (stages_t, extents_t))>(
                _impl::exec_stage_f<extent_t, ItDomain>{it_domain});
        }
    };

    template <template <uint_t> class Functor, class Extent, class Args, class LocationType, size_t RepeatFactor>
    struct all_colors_stage {
        GRIDTOOLS_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_arg, Args>::value), GT_INTERNAL_ERROR);

        using extent_t = Extent;

        template <uint_t>
        struct contains_color : std::true_type {};

        template <uint_t Color, class ItDomain>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            using eval_t = typename get_iterate_domain_remapper<ItDomain, Args, LocationType, Color>::type;
            eval_t eval{it_domain};
            gridtools::for_each<GT_META_CALL(meta::make_indices_c, RepeatFactor)>(
                _impl::call_do<Functor<Color>, eval_t>{&eval});
        }

        template <class ItDomain>
        struct exec_for_color_f {
            ItDomain &m_domain;
            template <class Color>
            GT_FUNCTION void operator()(Color) const {
                exec<Color::value>(m_domain);
                m_domain.increment_c();
            }
        };

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            static constexpr auto n_colors = LocationType::n_colors::value;
            gridtools::for_each<GT_META_CALL(meta::make_indices_c, n_colors)>(exec_for_color_f<ItDomain>{it_domain});
            it_domain.template increment_c<-n_colors>();
        }
    };

    template <uint_t Color, class Functor, class Extent, class Args, class LocationType, size_t RepeatFactor>
    struct color_specific_stage {
        GRIDTOOLS_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_arg, Args>::value), GT_INTERNAL_ERROR);

        using extent_t = Extent;

        template <uint_t C>
        struct contains_color : bool_constant<C == Color> {};

        template <uint_t C, class ItDomain, enable_if_t<C == Color, int> = 0>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            using eval_t = typename get_iterate_domain_remapper<ItDomain, Args, LocationType, Color>::type;
            eval_t eval{it_domain};
            gridtools::for_each<GT_META_CALL(meta::make_indices_c, RepeatFactor)>(
                _impl::call_do<Functor, eval_t>{&eval});
        }

        template <uint_t C, class ItDomain, enable_if_t<C != Color, int> = 0>
        static GT_FUNCTION void exec(ItDomain &it_domain) {}

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            it_domain.template increment_c<Color>();
            exec<Color>(it_domain);
            it_domain.template increment_c<-Color>();
        }
    };

    template <size_t Color>
    struct stage_contains_color {
        template <class Stage>
        struct apply : Stage::template contains_color<Color> {};
    };

} // namespace gridtools
