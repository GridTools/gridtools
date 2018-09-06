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
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/generic_metafunctions/meta.hpp"
#include "../../common/generic_metafunctions/type_traits.hpp"
#include "../../common/host_device.hpp"
#include "../arg.hpp"
#include "../extent.hpp"
#include "../hasdo.hpp"
#include "../location_type.hpp"
#include "./iterate_domain_expandable_parameters.hpp"
#include "./iterate_domain_remapper.hpp"

/**
 *   @file
 *
 *   Stage concept represents elementary functor from the backend implementor point of view.
 *   Stage concept for icosahedral grid is defined similar as for structured grid (with some additions)
 *
 *   [exactly the same as for strgrid]
 *   Stage must have the nested `extent_t` type or an alias that has to model Extent concept.
 *   The meaning: the stage should be computed in the area that is extended from the user provided computation aria by
 *   that much.
 *
 *   [almost the same as for strgrid]
 *   Stage also have static `exec` method that accepts an object by reference that models IteratorDomain.
 *   `exec` should execute an elementary functor for all colors from the grid point that IteratorDomain points to.
 *   precondition: IteratorDomain should point to the first color.
 *   postcondition: IteratorDomain still points to the first color.
 *
 *   [icgrid specific]
 *   Stage has templated variation of `exec` which accept color number as a first template parameter. This variation
 *   does not iterate on colors; it executes an elementary functor for the given color.
 *   precondition: IteratorDomain should point to the same color as one in exec parameter.
 *
 *   [icgrid specific]
 *   Stage has netsted metafunction contains_color<Color> that evaluates to std::false_type if for the given color
 *   the elementary function is not executed.
 *
 *   Note that the Stage is (and should stay) backend independent. The core of gridtools passes stages [split by k-loop
 *   intervals and independent groups] to the backend in the form of compile time only parameters.
 *
 *   TODO(anstaf): add `is_stage<T>` trait
 */

namespace gridtools {

    namespace _impl {
        template <class Functor, class Eval>
        struct call_do_f {
            Eval *m_eval;
            template <class Index>
            GT_FUNCTION void operator()() const {
                using eval_t = iterate_domain_expandable_parameters<Eval, Index::value + 1>;
                Functor::template Do<eval_t &>(*reinterpret_cast<eval_t *>(m_eval));
            }
        };

        template <class T>
        GT_META_DEFINE_ALIAS(functor_or_void, bool_constant, has_do<T>::value || std::is_void<T>::value);

    } // namespace _impl

    /**
     *   A stage that is produced from the icgrid esf_description data
     *
     * @tparam Functors - a list of elementary functors (with the intervals already bound). The position in the list
     *                    corresponds to the color number. If a functor should not be executed for the given color,
     *                    the correspondent element in the list is `void`
     */
    template <class Functors, class Extent, class Args, class LocationType, size_t RepeatFactor>
    struct stage {
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<_impl::functor_or_void, Functors>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_arg, Args>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(is_location_type<LocationType>::value, GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT(meta::length<Functors>::value == LocationType::n_colors::value, GT_INTERNAL_ERROR);

        using extent_t = Extent;

        template <uint_t Color, class Functor = GT_META_CALL(meta::at_c, (Functors, Color))>
        struct contains_color : bool_constant<!std::is_void<Functor>::value> {};

        template <uint_t Color, class ItDomain, enable_if_t<contains_color<Color>::value, int> = 0>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            using eval_t = typename get_iterate_domain_remapper<ItDomain, Args, LocationType, Color>::type;
            eval_t eval{it_domain};
            for_each_type<GT_META_CALL(meta::make_indices_c, RepeatFactor)>(
                _impl::call_do_f<GT_META_CALL(meta::at_c, (Functors, Color)), eval_t>{&eval});
        }

        template <uint_t Color, class ItDomain, enable_if_t<!contains_color<Color>::value, int> = 0>
        static GT_FUNCTION void exec(ItDomain &it_domain) {}

        template <class ItDomain>
        struct exec_for_color_f {
            ItDomain &m_domain;
            template <class Color>
            GT_FUNCTION void operator()() const {
                exec<Color::value>(m_domain);
                m_domain.increment_c();
            }
        };

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            static constexpr auto n_colors = LocationType::n_colors::value;
            for_each_type<GT_META_CALL(meta::make_indices_c, n_colors)>(exec_for_color_f<ItDomain>{it_domain});
            it_domain.template increment_c<-n_colors>();
        }
    };

    template <size_t Color>
    struct stage_contains_color {
        template <class Stage>
        struct apply : Stage::template contains_color<Color> {};
    };

    template <size_t Color>
    struct stage_group_contains_color {
        template <class Stages>
        GT_META_DEFINE_ALIAS(apply, meta::any_of, (stage_contains_color<Color>::template apply, Stages));
    };

} // namespace gridtools
