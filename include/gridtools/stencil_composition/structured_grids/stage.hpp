/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 *   @file
 *
 *   Stage concept represents elementary functor from the backend implementor point of view.
 *
 *   Stage must have the nested `extent_t` type or an alias that has to model Extent concept.
 *   The meaning: the stage should be computed in the area that is extended from the user provided computation area by
 *   that much.
 *
 *   Stage also have static `exec` method that accepts an object by reference that models IteratorDomain.
 *   `exec` should execute an elementary functor in the grid point that IteratorDomain points to.
 *
 *   Note that the Stage is (and should stay) backend independent. The core of gridtools passes stages [split by k-loop
 *   intervals and independent groups] to the backend in the form of compile time only parameters.
 *
 *   TODO(anstaf): add `is_stage<T>` trait
 */

#pragma once

#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/host_device.hpp"
#include "../../meta.hpp"
#include "../accessor_intent.hpp"
#include "../arg.hpp"
#include "../expressions/expr_base.hpp"
#include "../has_apply.hpp"
#include "../iterate_domain_fwd.hpp"
#include "../positional.hpp"
#include "../sid/multi_shift.hpp"
#include "dim.hpp"
#include "extent.hpp"

namespace gridtools {

    namespace stage_impl_ {
        template <class ItDomain, class Args>
        struct itdomain_evaluator {
            GT_STATIC_ASSERT((meta::all_of<is_plh, Args>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);

            ItDomain const &m_it_domain;

            template <class Accessor>
            GT_FUNCTION decltype(auto) operator()(Accessor const &arg) const {
                return apply_intent<Accessor::intent_v>(
                    m_it_domain.template deref<meta::at_c<Args, Accessor::index_t::value>>(arg));
            }

            template <class Op, class... Ts>
            GT_FUNCTION auto operator()(expr<Op, Ts...> const &arg) const {
                return expressions::evaluation::value(*this, arg);
            }

            GT_FUNCTION int_t i() const { return m_it_domain.i(); }
            GT_FUNCTION int_t j() const { return m_it_domain.j(); }
            GT_FUNCTION int_t k() const { return m_it_domain.k(); }
        };

        struct default_deref_f {
            template <class Arg, class T>
            GT_FUNCTION T &operator()(T *ptr) const {
                return *ptr;
            }
        };

        template <class Ptr, class Strides, class Args, class Deref>
        struct evaluator {
            Ptr const &m_ptr;
            Strides const &m_strides;

            template <class Arg>
            using ref_type =
                decltype(Deref{}.template operator()<Arg>(host_device::at_key<Arg>(std::declval<Ptr const &>())));

            template <class Accessor, class Arg = meta::at_c<Args, Accessor::index_t::value>>
            GT_FUNCTION apply_intent_t<Accessor::intent_v, ref_type<Arg>> operator()(Accessor const &acc) const {
                auto ptr = host_device::at_key<Arg>(m_ptr);
                sid::multi_shift<Arg>(ptr, m_strides, acc);
                return Deref{}.template operator()<Arg>(ptr);
            }

            template <class Op, class... Ts>
            GT_FUNCTION auto operator()(expr<Op, Ts...> const &arg) const {
                return expressions::evaluation::value(*this, arg);
            }

            GT_FUNCTION int_t i() const { return *host_device::at_key<positional<dim::i>>(m_ptr); }
            GT_FUNCTION int_t j() const { return *host_device::at_key<positional<dim::j>>(m_ptr); }
            GT_FUNCTION int_t k() const { return *host_device::at_key<positional<dim::k>>(m_ptr); }
        };
    } // namespace stage_impl_

    /**
     *   A stage that is associated with an elementary functor.
     */
    template <class Functor, class Extent, class Args>
    struct regular_stage {
        GT_STATIC_ASSERT(has_apply<Functor>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_extent<Extent>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_plh, Args>::value), GT_INTERNAL_ERROR);

        using extent_t = Extent;

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain const &it_domain) {
            GT_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);
            stage_impl_::itdomain_evaluator<ItDomain, Args> eval{it_domain};
            Functor::apply(eval);
        }

        template <class Deref = stage_impl_::default_deref_f, class Ptr, class Strides>
        GT_FUNCTION void operator()(Ptr const &ptr, Strides const &strides) const {
            stage_impl_::evaluator<Ptr, Strides, Args, Deref> eval{ptr, strides};
            Functor::apply(eval);
        }
    };

    template <class Stage, class... Stages>
    struct compound_stage {
        using type = compound_stage;

        using extent_t = typename Stage::extent_t;

        GT_STATIC_ASSERT(sizeof...(Stages) != 0, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((conjunction<std::is_same<typename Stages::extent_t, extent_t>...>::value), GT_INTERNAL_ERROR);

        template <class ItDomain>
        static GT_FUNCTION void exec(ItDomain const &it_domain) {
            GT_STATIC_ASSERT(is_iterate_domain<ItDomain>::value, GT_INTERNAL_ERROR);
            Stage::exec(it_domain);
            (void)(int[]){(Stages::exec(it_domain), 0)...};
        }

        template <class Deref = stage_impl_::default_deref_f, class Ptr, class Strides>
        GT_FUNCTION void operator()(Ptr const &ptr, Strides const &strides) const {
            Stage{}.template operator()<Deref>(ptr, strides);
            (void)(int[]){(Stages{}.template operator()<Deref>(ptr, strides), 0)...};
        }
    };
} // namespace gridtools
