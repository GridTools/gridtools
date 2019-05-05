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
#include "../../meta.hpp"
#include "../accessor_intent.hpp"
#include "../arg.hpp"
#include "../extent.hpp"
#include "../has_apply.hpp"
#include "../iterate_domain_fwd.hpp"
#include "../location_type.hpp"
#include "../sid/multi_shift.hpp"
#include "dim.hpp"
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
        GT_META_DEFINE_ALIAS(functor_or_void, bool_constant, has_apply<T>::value || std::is_void<T>::value);

        template <class ItDomain, class Args, class LocationType, uint_t Color>
        struct itdomain_evaluator {
            GT_STATIC_ASSERT((is_iterate_domain<ItDomain>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((meta::all_of<is_plh, Args>::value), GT_INTERNAL_ERROR);

            ItDomain const &m_it_domain;

            template <class Accessor>
            GT_FUNCTION auto operator()(Accessor const &acc) const GT_AUTO_RETURN(apply_intent<Accessor::intent_v>(
                m_it_domain.template deref<GT_META_CALL(meta::at_c, (Args, Accessor::index_t::value))>(acc)));

            template <class ValueType, class LocationTypeT, class Reduction, class... Accessors>
            GT_FUNCTION ValueType operator()(
                on_neighbors<ValueType, LocationTypeT, Reduction, Accessors...> onneighbors) const {
                constexpr auto offsets = connectivity<LocationType, LocationTypeT, Color>::offsets();
                for (auto &&offset : offsets)
                    onneighbors.m_value = onneighbors.m_function(
                        apply_intent<intent::in>(
                            m_it_domain.template deref<GT_META_CALL(meta::at_c, (Args, Accessors::index_t::value))>(
                                offset))...,
                        onneighbors.m_value);
                return onneighbors.m_value;
            }
        };

        struct default_dereference_f {
            template <class T>
            GT_FUNCTION T &operator()(T *ptr) const {
                return *ptr;
            }
        };

        template <class Ptr, class Strides, class Args, class Deref, class LocationType, uint_t Color>
        struct evaluator {
            Ptr const &m_ptr;
            Strides const &m_strides;

            template <class Arg, class Accessor>
            GT_FUNCTION auto get_ptr(Accessor const &acc) const -> decay_t<decltype(host_device::at_key<Arg>(m_ptr))> {
                auto res = host_device::at_key<Arg>(m_ptr);
                sid::multi_shift<Arg>(res, m_strides, acc);
                return res;
            }

            template <class Accessor,
                class Offset,
                class Arg = GT_META_CALL(meta::at_c, (Args, Accessor::index_t::value))>
            GT_FUNCTION auto neighbor(Offset const &offset) const
                GT_AUTO_RETURN(apply_intent<intent::in>(typename Deref::template apply<Arg>{}(get_ptr<Arg>(offset))));

            template <class Accessor, class Arg = GT_META_CALL(meta::at_c, (Args, Accessor::index_t::value))>
            GT_FUNCTION auto operator()(Accessor const &acc) const GT_AUTO_RETURN(
                apply_intent<Accessor::intent_v>(typename Deref::template apply<Arg>{}(get_ptr<Arg>(acc))));

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
    template <class Functors, class Extent, class Args, class LocationType>
    struct stage {
        GT_STATIC_ASSERT((meta::all_of<stage_impl_::functor_or_void, Functors>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_plh, Args>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_location_type<LocationType>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(meta::length<Functors>::value == LocationType::n_colors::value, GT_INTERNAL_ERROR);

        using extent_t = Extent;
        using n_colors = typename LocationType::n_colors;

        template <uint_t Color, class Functor = GT_META_CALL(meta::at_c, (Functors, Color))>
        struct contains_color : bool_constant<!std::is_void<Functor>::value> {};

        template <uint_t Color, class ItDomain, enable_if_t<contains_color<Color>::value, int> = 0>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            using eval_t = stage_impl_::itdomain_evaluator<ItDomain, Args, LocationType, Color>;
            using functor_t = GT_META_CALL(meta::at_c, (Functors, Color));
            eval_t eval{it_domain};
            functor_t::apply(eval);
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
            static constexpr int_t n_colors = LocationType::n_colors::value;
            host_device::for_each_type<GT_META_CALL(meta::make_indices_c, n_colors)>(
                exec_for_color_f<ItDomain>{it_domain});
            it_domain.increment_c(integral_constant<int_t, -n_colors>{});
        }

        template <uint_t Color, bool = contains_color<Color>::value>
        struct colored_stage {
            template <class Ptr, class Strides, class Deref = meta::always<stage_impl_::default_dereference_f>>
            GT_FUNCTION void operator()(Ptr const &ptr, Strides const &strides, Deref = {}) const {
                using eval_t = stage_impl_::evaluator<Ptr, Strides, Args, Deref, LocationType, Color>;
                using functor_t = GT_META_CALL(meta::at_c, (Functors, Color));
                functor_t::template apply<eval_t const &>(eval_t{ptr, strides});
            }
        };

        template <uint_t Color>
        struct colored_stage<Color, false> {
            template <class Ptr, class Strides, class Deref = meta::always<stage_impl_::default_dereference_f>>
            GT_FUNCTION void operator()(Ptr const &, Strides const &, Deref = {}) const {}
        };

        template <class Ptr, class Strides, class Deref>
        struct call_for_color_f {
            Ptr &m_ptr;
            Strides const &m_strides;
            template <class Color>
            GT_FUNCTION void operator()() const {
                colored_stage<Color::value>{}(m_ptr, m_strides, Deref{});
                sid::shift(m_ptr, sid::get_stride<dim::c>(m_strides), integral_constant<int_t, 1>{});
            }
        };

        template <class Ptr, class Strides, class Deref = meta::always<stage_impl_::default_dereference_f>>
        GT_FUNCTION void operator()(Ptr &ptr, Strides const &strides, Deref = {}) const {
            static constexpr int_t n_colors = LocationType::n_colors::value;
            host_device::for_each_type<GT_META_CALL(meta::make_indices_c, n_colors)>(
                call_for_color_f<Ptr, Strides, Deref>{ptr, strides});
            sid::shift(ptr, sid::get_stride<dim::c>(strides), integral_constant<int_t, -n_colors>{});
        }
    };

    template <class Stage, class... Stages>
    struct compound_stage {
        using extent_t = typename Stage::extent_t;
        using n_colors = typename Stage::n_colors;

        GT_STATIC_ASSERT(sizeof...(Stages) != 0, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((conjunction<std::is_same<typename Stages::extent_t, extent_t>...>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((conjunction<std::is_same<typename Stages::n_colors, n_colors>...>::value), GT_INTERNAL_ERROR);

        template <uint_t Color>
        struct contains_color : disjunction<typename Stage::template contains_color<Color>,
                                    typename Stages::template contains_color<Color>...> {};

        template <uint_t Color, class ItDomain, enable_if_t<contains_color<Color>::value, int> = 0>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            Stage::template exec<Color>(it_domain);
            (void)(int[]){(Stages::template exec<Color>(it_domain), 0)...};
        }

        template <uint_t Color, class ItDomain, enable_if_t<!contains_color<Color>::value, int> = 0>
        static GT_FUNCTION void exec(ItDomain &it_domain) {}

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain &it_domain) {
            GT_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);
            Stage::exec(it_domain);
            (void)(int[]){(Stages::exec(it_domain), 0)...};
        }

        template <uint_t Color, bool = contains_color<Color>::value>
        struct colored_stage {
            template <class Ptr, class Strides, class Deref = meta::always<stage_impl_::default_dereference_f>>
            GT_FUNCTION void operator()(Ptr &ptr, Strides const &strides, Deref = {}) const {
                typename Stage::template colored_stage<Color>{}(ptr, strides);
                (void)(int[]){(typename Stages::template colored_stage<Color>{}(ptr, strides), 0)...};
            }
        };

        template <uint_t Color>
        struct colored_stage<Color, false> {
            template <class Ptr, class Strides, class Deref = meta::always<stage_impl_::default_dereference_f>>
            GT_FUNCTION void operator()(Ptr &, Strides const &, Deref = {}) const {}
        };

        template <class Ptr, class Strides, class Deref = meta::always<stage_impl_::default_dereference_f>>
        GT_FUNCTION void operator()(Ptr &ptr, Strides const &strides, Deref deref = {}) const {
            Stage{}(ptr, strides, deref);
            (void)(int[]){(Stages{}(ptr, strides, deref), 0)...};
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
