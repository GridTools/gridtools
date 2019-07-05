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
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../meta.hpp"
#include "../accessor_intent.hpp"
#include "../arg.hpp"
#include "../extent.hpp"
#include "../has_apply.hpp"
#include "../iterate_domain_fwd.hpp"
#include "../location_type.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/multi_shift.hpp"
#include "dim.hpp"
#include "esf.hpp"
#include "icosahedral_topology.hpp"
#include "on_neighbors.hpp"

/**
 *   @file
 *
 *   Stage concept represents elementary functor from the backend implementor point of view.
 *   Stage concept for icosahedral grid is defined similar as for structured grid (with some additions)
 *
 *   Stage must have the nested `extent_t` type or an alias that has to model Extent concept.
 *   The meaning: the stage should be computed in the area that is extended from the user provided computation area by
 *   that much.
 *
 *   Stage also have static `exec` method that accepts an object by reference that models IteratorDomain.
 *   `exec` should execute an elementary functor for all colors from the grid point that IteratorDomain points to.
 *   precondition: IteratorDomain should point to the first color.
 *   postcondition: IteratorDomain still points to the first color.
 *
 *   Stage has templated variation of `exec` which accept color number as a first template parameter. This variation
 *   does not iterate on colors; it executes an elementary functor for the given color.
 *   precondition: IteratorDomain should point to the same color as one in exec parameter.
 *
 *   Stage has netsted metafunction contains_color<Color> that evaluates to std::false_type if for the given color
 *   the elementary function is not executed.
 *
 *   Note that the Stage is (and should stay) backend independent. The core of gridtools passes stages [split by k-loop
 *   intervals and independent groups] to the backend in the form of compile time only parameters.
 *
 *   TODO(anstaf): add `is_stage<T>` trait
 */

namespace gridtools {

    namespace stage_impl_ {
        template <class T>
        using functor_or_void = bool_constant<has_apply<T>::value || std::is_void<T>::value>;

        template <class ItDomain, class Args, class LocationType, uint_t Color>
        struct itdomain_evaluator {
            GT_STATIC_ASSERT((is_iterate_domain<ItDomain>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((meta::all_of<is_plh, Args>::value), GT_INTERNAL_ERROR);

            ItDomain const &m_it_domain;

            template <class Accessor>
            GT_FUNCTION decltype(auto) operator()(Accessor const &acc) const {
                return apply_intent<Accessor::intent_v>(
                    m_it_domain.template deref<meta::at_c<Args, Accessor::index_t::value>>(acc));
            }

            template <class ValueType, class LocationTypeT, class Reduction, class... Accessors>
            GT_FUNCTION ValueType operator()(
                on_neighbors<ValueType, LocationTypeT, Reduction, Accessors...> onneighbors) const {
                constexpr auto offsets = connectivity<LocationType, LocationTypeT, Color>::offsets();
                for (auto &&offset : offsets)
                    onneighbors.m_value = onneighbors.m_function(
                        apply_intent<intent::in>(
                            m_it_domain.template deref<meta::at_c<Args, Accessors::index_t::value>>(offset))...,
                        onneighbors.m_value);
                return onneighbors.m_value;
            }
        };

        struct default_deref_f {
            template <class Arg, class T>
            GT_FUNCTION T &operator()(T *ptr) const {
                return *ptr;
            }
        };

        template <class Ptr, class Strides, class Args, class Deref, class LocationType, uint_t Color>
        struct evaluator {
            Ptr const &m_ptr;
            Strides const &m_strides;

            template <class Arg>
            using ref_type =
                decltype(Deref{}.template operator()<Arg>(host_device::at_key<Arg>(std::declval<Ptr const &>())));

            template <class Arg, class Offset>
            GT_FUNCTION ref_type<Arg> get_ref(Offset const &offset) const {
                auto ptr = host_device::at_key<Arg>(m_ptr);
                sid::multi_shift<Arg>(ptr, m_strides, offset);
                return Deref{}.template operator()<Arg>(ptr);
            }

            template <class Accessor, class Arg = meta::at_c<Args, Accessor::index_t::value>>
            GT_FUNCTION apply_intent_t<Accessor::intent_v, ref_type<Arg>> operator()(Accessor const &acc) const {
                return get_ref<Arg>(acc);
            }

            template <class Accessor, class Offset, class Arg = meta::at_c<Args, Accessor::index_t::value>>
            GT_FUNCTION apply_intent_t<intent::in, ref_type<Arg>> neighbor(Offset const &offset) const {
                return get_ref<Arg>(offset);
            }

            template <class ValueType, class LocationTypeT, class Reduction, class... Accessors>
            GT_FUNCTION ValueType operator()(
                on_neighbors<ValueType, LocationTypeT, Reduction, Accessors...> onneighbors) const {
                static constexpr auto offsets = connectivity<LocationType, LocationTypeT, Color>::offsets();
                for (auto &&offset : offsets)
                    onneighbors.m_value = onneighbors.m_function(neighbor<Accessors>(offset)..., onneighbors.m_value);
                return onneighbors.m_value;
            }
        };

    } // namespace stage_impl_

    /**
     *   A stage that is produced from the icgrid esf_description data
     *
     * @tparam Functors - a list of elementary functors (with the intervals already bound). The position in the list
     *                    corresponds to the color number. If a functor should not be executed for the given color,
     *                    the correspondent element in the list is `void`
     */
    template <class Functors, class Extent, class Esf>
    struct stage {
        GT_STATIC_ASSERT((meta::all_of<stage_impl_::functor_or_void, Functors>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_esf_descriptor<Esf>::value, GT_INTERNAL_ERROR);

        using esf_t = Esf;
        using location_type = typename Esf::location_type;
        using n_colors = typename location_type::n_colors;
        using args_t = typename Esf::args_t;

        GT_STATIC_ASSERT(meta::length<Functors>::value == n_colors::value, GT_INTERNAL_ERROR);

        using extent_t = Extent;

        static GT_FUNCTION Extent extent() { return {}; }

        template <uint_t Color, class Functor = meta::at_c<Functors, Color>>
        struct contains_color : bool_constant<!std::is_void<Functor>::value> {};

        template <uint_t Color, class ItDomain, std::enable_if_t<contains_color<Color>::value, int> = 0>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            using eval_t = stage_impl_::itdomain_evaluator<ItDomain, args_t, location_type, Color>;
            using functor_t = meta::at_c<Functors, Color>;
            eval_t eval{it_domain};
            functor_t::apply(eval);
        }

        template <uint_t Color, class ItDomain, std::enable_if_t<!contains_color<Color>::value, int> = 0>
        static GT_FUNCTION void exec(ItDomain &it_domain) {}

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            host_device::for_each<meta::make_indices<n_colors>>([&](auto color) {
                exec<decltype(color)::value>(it_domain);
                it_domain.increment_c();
            });
            it_domain.increment_c(integral_constant<int_t, -(int_t)n_colors::value>{});
        }

        template <uint_t Color, bool = contains_color<Color>::value>
        struct colored_stage {
            template <class Deref = stage_impl_::default_deref_f, class Ptr, class Strides>
            GT_FUNCTION void operator()(Ptr const &ptr, Strides const &strides) const {
                using eval_t = stage_impl_::evaluator<Ptr, Strides, args_t, Deref, location_type, Color>;
                using functor_t = meta::at_c<Functors, Color>;
                functor_t::template apply<eval_t const &>(eval_t{ptr, strides});
            }
        };

        template <uint_t Color>
        struct colored_stage<Color, false> {
            template <class Deref = stage_impl_::default_deref_f, class Ptr, class Strides>
            GT_FUNCTION void operator()(Ptr const &, Strides const &) const {}
        };

        template <class Deref = stage_impl_::default_deref_f, class Ptr, class Strides>
        GT_FUNCTION void operator()(Ptr ptr, Strides const &strides) const {
            host_device::for_each<meta::make_indices<n_colors>>([&](auto color) {
                colored_stage<decltype(color)::value>{}.template operator()<Deref>(ptr, strides);
                sid::shift(ptr, sid::get_stride<dim::c>(strides), integral_constant<int_t, 1>());
            });
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
        using apply = meta::any_of<stage_contains_color<Color>::template apply, Stages>;
    };
} // namespace gridtools
